/*
 * This file is part of the Substack Analysis (SUSAN) framework.
 * Copyright (c) 2018-2021 Ricardo Miguel Sanchez Loayza.
 * Max Planck Institute of Biophysics
 * Department of Structural Biology - Kudryashev Group.
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef CTF_LINEARIZER_H
#define CTF_LINEARIZER_H

#include "datatypes.h"
#include "particles.h"
#include "tomogram.h"
#include "stack_reader.h"
#include "gpu.h"
#include "gpu_fft.h"
#include "gpu_kernel.h"
#include "gpu_kernel_ctf.h"
#include "substack_crop.h"
#include "mrc.h"
#include "io.h"
#include "points_provider.h"
#include "svg.h"
#include "reconstruct_args.h"
#include <algorithm>
#include <iostream>

class CtfLinearizer{

protected:
    class Normal {
    public:
        int3 ss_c;
        dim3 blk;
        dim3 grd_c;
        GPU::GArrSingle ss_foreground;
        GPU::GArrSingle ss_fg_masked;
        long numel;

        Normal(int m, int n, int k) {
            numel = m*n*k;
            ss_c  = make_int3(m,n,k);
            blk   = GPU::get_block_size_2D();
            grd_c = GPU::calc_grid_size(blk,ss_c.x,ss_c.y,ss_c.z);

            ss_foreground.alloc(m*n*k);
            ss_fg_masked.alloc(m*n*k);
        }

        void load_rmv_bg_msk(const float*cpu_input,float2 fpix_range) {
            GPU::GArrSingle3 ss_filter;
            uint32           n_filter;
            create_filter(n_filter,ss_filter);

            GPU::GTex2DSingle ss_lin;
            ss_lin.alloc(ss_c.x,ss_c.y,ss_c.z);

            /// use ss_fg_masked as an upload buffer:
            cudaMemcpy( (void*)ss_fg_masked.ptr, (const void*)(cpu_input), sizeof(float)*numel, cudaMemcpyHostToDevice);
            GpuKernelsCtf::rmv_bg<<<grd_c,blk>>>(ss_foreground.ptr,ss_fg_masked.ptr,ss_filter.ptr,n_filter,ss_c);
            GpuKernels::load_surf<<<grd_c,blk>>>(ss_lin.surface,ss_foreground.ptr,ss_c);
            GpuKernelsCtf::tangential_blur<<<grd_c,blk>>>(ss_foreground.ptr,ss_lin.texture,ss_c);
            GpuKernels::conv_gaussian<<<grd_c,blk>>>(ss_fg_masked.ptr,ss_foreground.ptr,0.1250,23.9907,ss_c);
            cudaMemcpy( (void*)ss_foreground.ptr, (const void*)(ss_fg_masked.ptr), sizeof(float)*numel, cudaMemcpyDeviceToDevice);

            GpuKernelsCtf::keep_fpix_range<<<grd_c,blk>>>(ss_fg_masked.ptr,fpix_range,ss_c);
        }

    protected:
        static void create_filter(uint32&n_filter,GPU::GArrSingle3&ss_filter) {
            Vec3*c_filt = PointsProvider::circle(n_filter,10,10);
            ss_filter.alloc(n_filter);
            for(uint32 i=0;i<n_filter;i++) {
                float R2 = (c_filt[i].x*c_filt[i].x) + (c_filt[i].y*c_filt[i].y);
                float R4 = R2*R2;
                // h = firls(20,[0 1/16 1/16 1],[1 1 0 0]);
                // [X,Y] = meshgrid(-10:10,-10:10);
                // R = sqrt( X.*X + Y.*Y )+11;
                // hh = interp1(h,R,'linear',0);
                // hh = hh/sum(hh(:));
                // % From quadratic fit: hh = 4.7764e-08*R.^4 -2.8131e-05*R.^2 0.0044146
                c_filt[i].z = 0.0044146 + 4.7764e-08*R4 - 2.8131e-05*R2;
            }
            cudaMemcpy( (void*)ss_filter.ptr, (const void*)(c_filt), sizeof(Vec3)*n_filter, cudaMemcpyHostToDevice);
            delete [] c_filt;
        }
    };

    class Linear {
    public:
        float4*c_def;
        int3 ss_c;
        int3 ss_r;
        dim3 blk;
        dim3 grd_c;
        dim3 grd_r;
        GPU::GArrSingle  r_linear_in;
        GPU::GArrSingle  r_lin_expnd;
        GPU::GArrSingle  r_lin_mskd;
        GPU::GArrSingle  ss_pow_spct;
        GPU::GArrSingle  ss_ps_sum_z;
        GPU::GArrSingle2 c_linear;
        GPU::GArrSingle4 g_def;

        GpuFFT::FFT2D     fft2;
        GPU::GTex2DSingle ss_lin;

        Linear(int m, int n, int k) {
            ss_c  = make_int3(m,n,k);
            ss_r  = make_int3(n,n,k);
            blk   = GPU::get_block_size_2D();
            grd_c = GPU::calc_grid_size(blk,ss_c.x,ss_c.y,ss_c.z);
            grd_r = GPU::calc_grid_size(blk,ss_r.x,ss_r.y,ss_r.z);

            r_linear_in.alloc(m*n*k);
            r_lin_expnd.alloc(n*n*k);
            r_lin_mskd.alloc(n*n*k);
            ss_pow_spct.alloc(m*n*k);
            ss_ps_sum_z.alloc(m*n);
            c_linear.alloc(m*n*k);
            g_def.alloc(k);

            ss_lin.alloc(m,n,k);
            fft2.alloc(m,n,k);

            c_def = new float4[k];
        }

        ~Linear() {
            delete [] c_def;
        }

        void load_linearize(const float*gpu_input,float linearization_scale) {
            /// ss_pow_spct as input buffer
            GpuKernels::load_surf<<<grd_c,blk>>>(ss_lin.surface,gpu_input,ss_c);
            GpuKernelsCtf::ctf_linearize<<<grd_c,blk>>>(ss_pow_spct.ptr,ss_lin.texture,linearization_scale,ss_c);
            GpuKernels::conv_gaussian<<<grd_c,blk>>>(r_linear_in.ptr,ss_pow_spct.ptr,0.1250,23.9907,ss_c);
            GpuKernels::expand_ps_hermitian<<<grd_r,blk>>>(r_lin_expnd.ptr,r_linear_in.ptr,ss_r);
        }

        void mask_and_power_spectrum(const float2*gpu_fp_mask,const float2 def_lin_range) {
            GpuKernels::apply_circular_mask<<<grd_r,blk>>>(r_lin_mskd.ptr,r_lin_expnd.ptr,gpu_fp_mask,ss_r);
            GpuKernels::stk_scale<<<grd_r,blk>>>(r_lin_mskd.ptr,gpu_fp_mask,ss_r);
            GpuKernels::fftshift2D<<<grd_r,blk>>>(r_lin_mskd.ptr,ss_r);
            fft2.exec(c_linear.ptr,r_lin_mskd.ptr);
            GpuKernels::fftshift2D<<<grd_c,blk>>>(c_linear.ptr,ss_c);
            GpuKernels::load_surf_real<<<grd_c,blk>>>(ss_lin.surface,c_linear.ptr,ss_c);
            GpuKernelsCtf::tangential_blur<<<grd_c,blk>>>(ss_pow_spct.ptr,ss_lin.texture,ss_c);
            GpuKernels::load_surf<<<grd_c,blk>>>(ss_lin.surface,ss_pow_spct.ptr,ss_c);
            GpuKernelsCtf::radial_edge_detect<<<grd_c,blk>>>(ss_pow_spct.ptr,ss_lin.texture,ss_c);
            GpuKernels::load_surf<<<grd_c,blk>>>(ss_lin.surface,ss_pow_spct.ptr,ss_c);
            GpuKernelsCtf::radial_highpass<<<grd_c,blk>>>(ss_pow_spct.ptr,ss_lin.texture,ss_c);
            GpuKernelsCtf::keep_fpix_range<<<grd_c,blk>>>(ss_pow_spct.ptr,def_lin_range,ss_c);
        }

        void sum_z() {
            GpuKernelsCtf::sum_along_z<<<grd_c,blk>>>(ss_ps_sum_z.ptr,ss_pow_spct.ptr,ss_c);
        }

        void load_def_range(float def_u_fpix,float def_v_fpix,float def_ang,float tlt_fpix,Tomogram*p_tomo) {
            for(int k=0;k<ss_c.z;k++) {
                V3f euZYZ;
                Math::Rmat_eZYZ(euZYZ,p_tomo->R[k]);
                c_def[k].x = def_u_fpix;
                c_def[k].y = def_v_fpix;
                c_def[k].z = def_ang;
                c_def[k].w = tlt_fpix*(2-abs(cos(euZYZ(1))));
            }
            cudaMemcpy( (void*)g_def.ptr, (const void*)(c_def), sizeof(float4)*ss_c.z, cudaMemcpyHostToDevice);
        }

        void apply_def_range() {
            GpuKernelsCtf::mask_ellipsoid<<<grd_c,blk>>>(ss_pow_spct.ptr,g_def.ptr,ss_c);
        }

    };

    class FitEllipse {
    public:
        Eigen::MatrixXf points;
        Eigen::MatrixXf points_work;
        int n_pts;

        FitEllipse(int num_points) {
            n_pts = num_points;
            points = Eigen::MatrixXf::Zero(2,n_pts);
        }

        void fit(float&def_u,float&def_v,float&def_ang,const float*cpu_frame,const int M, const int N) {
            float x_c = 0;
            float y_c = N/2;

            for(int n_p=0;n_p<n_pts;n_p++) {

                float ang = n_p;
                ang *= (M_PI/n_pts);
                ang = ang - (M_PI/2);

                float x_v = cos(ang);
                float y_v = sin(ang);

                float max_val=0,max_x=0,max_y=0;

                for(float r=0;r<M-3;r+=1.0) {
                    float x = x_c + r*x_v;
                    float y = y_c + r*y_v;
                    int xx = (int)round(x);
                    int yy = (int)round(y);
                    float val = cpu_frame[ (int)xx + yy*(int)M ];
                    if( val > max_val ) {
                        max_val = val;
                        max_x = x;
                        max_y = y-y_c;
                    }
                }
                points(0,n_p)=max_x;
                points(1,n_p)=max_y;
            }

            exec_fit(def_u,def_v,def_ang);
        }

        void save_points(const char*name,const char*out_dir) {
            char filename[SUSAN_FILENAME_LENGTH];
            sprintf(filename,"%s/%s",out_dir,name);
            FILE*fp = fopen(filename,"w");
            fprintf(fp,"points = [ ...\n");
            for(int i=0;i<points_work.cols();i++) {
                 fprintf(fp,"\t%8.2f, %8.2f;...\n",points_work(0,i),points_work(1,i));
            }
            fprintf(fp,"];\n");
            fclose(fp);
        }

    protected:
        void exec_fit(float&def_u,float&def_v,float&def_ang) {

            float *radius = new float[n_pts];
            float *sorted = new float[n_pts];

            for(int i=0;i<points.cols();i++) {
                 radius[i] = sqrt(points(0,i)*points(0,i)+points(1,i)*points(1,i));
                 sorted[i] = radius[i];
            }

            Math::sort(sorted,points.cols());

            float r_median = sorted[(int)round(points.cols()/2)];
            float r_std = 0;

            for(int i=0;i<points.cols();i++) {
                float tmp = radius[i]-r_median;
                r_std += (tmp*tmp);
            }
            r_std = sqrt(r_std)/points.cols();

            r_std = max(r_std,1.75);
            float r_min = r_median-2*r_std;
            float r_max = r_median+2*r_std;

            int count = 0;
            for(int i=0;i<points.cols();i++) {
                if(radius[i] > r_min && radius[i] < r_max ) {
                    count++;
                }
            }

            int j=0;
            points_work = Eigen::MatrixXf::Zero(2,count);
            for(int i=0;i<points.cols();i++) {
                if(radius[i] > r_min && radius[i] < r_max ) {
                    points_work(0,j)=points(0,i);
                    points_work(1,j)=points(1,i);
                    j++;
                }
            }

            Math::fit_ellipsoid(def_u,def_v,def_ang,points_work);

            delete [] radius;
            delete [] sorted;
        }

    };

    class RadialAvgr {
    public:
        int3 ss_c;
        int3 ss_2;
        dim3 blk;
        dim3 grd_c;
        dim3 grd_2;
        GPU::GArrSingle   rad_normal;
        GPU::GTex2DSingle ss_raw;
        GPU::GArrSingle   rad_avg;
        GPU::GArrSingle   rad_env;
        GPU::GArrSingle2  rad_fourier;
        GPU::GArrSingle2  rad_hilbert;

        GpuFFT::FFT1D_full  fft1fwd;
        GpuFFT::IFFT1D_full fft1inv;

        long numel;
        float apix;
        float ix2def;
        float pi_lambda;

        RadialAvgr(int m, int n, int k,float in_ix2def,float in_pi_lambda,float in_apix) {
            apix = in_apix;
            ix2def = in_ix2def;
            pi_lambda = in_pi_lambda;

            numel = m*n*k;
            ss_c  = make_int3(m,n,k);
            ss_2  = make_int3(n,k,1);
            blk   = GPU::get_block_size_2D();
            grd_c = GPU::calc_grid_size(blk,ss_c.x,ss_c.y,ss_c.z);
            grd_2 = GPU::calc_grid_size(blk,ss_2.x,ss_2.y,ss_2.z);

            rad_normal.alloc(m*n*k);
            ss_raw.alloc(m,n,k);
            rad_avg.alloc(n*k);
            rad_env.alloc(n*k);
            rad_fourier.alloc(n*k);
            rad_hilbert.alloc(n*k);
            fft1fwd.alloc(n,k);
            fft1inv.alloc(n,k);
        }

        void load_and_radial_normalize(const float*gpu_in,const float4*gpu_def_in_fp) {
            GpuKernels::load_surf<<<grd_c,blk>>>(ss_raw.surface,gpu_in,ss_c);
            GpuKernelsCtf::ctf_radial_normalize<<<grd_c,blk>>>(rad_normal.ptr,ss_raw.texture,gpu_def_in_fp,ix2def,pi_lambda,apix,ss_c);
        }

        void calc_rad_average(float*cpu_rad_avg) {
            rad_avg.clear();
            rad_env.clear();
            GpuKernels::radial_ps_avg_double_side<<<grd_c,blk>>>(rad_avg.ptr,rad_env.ptr,rad_normal.ptr,ss_c);
            GpuKernels::divide<<<grd_2,blk>>>(rad_avg.ptr,rad_env.ptr,ss_2);

            int m = ss_c.x;
            int n = ss_c.y;
            int k = ss_c.z;

            cudaMemcpy((void*)cpu_rad_avg, (const void*)rad_avg.ptr, sizeof(float)*n*k, cudaMemcpyDeviceToHost);
            float avg,std,tmp;
            float *work;
            for(int j=0;j<k;j++) {
                avg = 0;
                std = 0;
                work = cpu_rad_avg + j*n;
                for(int i=0;i<m;i++) {
                    avg += work[ i ];
                }
                avg = avg/m;
                for(int i=0;i<m;i++) {
                    tmp = work[ i ] - avg;
                    std += (tmp*tmp);
                }
                std = sqrt(std/m);

                float th_min = avg - std;
                float th_max = avg + std;

                avg = 0;
                std = 0;
                for(int i=0;i<m;i++) {
                    if( work[ i ] > th_min && work[ i ] < th_max ) {
                        avg += work[ i ];
                    }
                }
                avg = avg/m;
                for(int i=0;i<m;i++) {
                    if( work[ i ] > th_min && work[ i ] < th_max ) {
                        tmp = work[ i ] - avg;
                        std += (tmp*tmp);
                    }
                }
                std = sqrt(std/m);

                for(int i=0;i<n;i++) {
                    work[ i ] = min( max( work[ i ] , avg-2.5*std ), avg+2.5*std );
                }
            }

        }

        void calc_env(float*cpu_rad_env,const float*cpu_rad_avg) {
            cudaMemcpy((void*)rad_avg.ptr, (const void*)cpu_rad_avg, sizeof(float)*ss_c.y*ss_c.z, cudaMemcpyHostToDevice);
            fft1fwd.exec(rad_fourier.ptr,rad_avg.ptr);
            GpuKernelsCtf::prepare_hilbert<<<grd_2,blk>>>(rad_fourier.ptr,ss_c.y/2,ss_2);
            fft1inv.exec(rad_hilbert.ptr,rad_fourier.ptr);
            GpuKernels::load_abs<<<grd_2,blk>>>(rad_env.ptr,rad_hilbert.ptr,ss_2);
            cudaMemcpy((void*)cpu_rad_env, (const void*)rad_env.ptr, sizeof(float)*ss_c.y*ss_c.z, cudaMemcpyDeviceToHost);
            for(int i=0;i<ss_c.y*ss_c.z;i++) {
                cpu_rad_env[i] = cpu_rad_env[i]/ss_c.y;
            }
        }
    };

    class FitCtf {
    public:
        float *ctf_signal;
        float *ctf_fitted;
        float *ctf_qualit;
        int N;
        int M;
        int K;

        float apix;
        float lambda_pi;

        float *s2;
        float *s4_lambda3_Cs_pi_2;
        float AC;
        float CA;

        FitCtf(int m, int n, int k,float in_apix,float in_lambda_pi,float in_lambda3_Cs_pi_2,float ac) {
            N = n;
            M = m;
            K = k;
            ctf_signal = new float[M*K];
            ctf_fitted = new float[M*K];
            ctf_qualit = new float[M*K];

            apix = in_apix;
            lambda_pi = in_lambda_pi;

            AC = ac;
            CA = sqrt(1-AC*AC);

            s2 = new float[M];
            s4_lambda3_Cs_pi_2 = new float[M];

            for(int i=0;i<M;i++) {
                s2[i] = (float)i;
                s2[i] = s2[i]/(N*apix);
                s2[i] = s2[i]*s2[i];
                s4_lambda3_Cs_pi_2[i] = s2[i]*s2[i]*in_lambda3_Cs_pi_2;
            }
        }

        ~FitCtf() {
            delete [] ctf_signal;
            delete [] ctf_fitted;
            delete [] ctf_qualit;
            delete [] s2;
            delete [] s4_lambda3_Cs_pi_2;
        }

        void load_ctf_avg(const float*ctf_avg,const float*ctf_env) {
            for(int j=0;j<K;j++) {
                for(int i=0;i<M;i++) {
                    ctf_signal[i + j*M] = ctf_avg[i + j*N]/ctf_env[i + j*N];
                    ctf_signal[i + j*M] = ctf_signal[i + j*M]/2 + 0.5;
                }
            }
        }

        void estimate(float4*ctf_info,float def_range,float def_step,float fpix_range_min,float fpix_range_max,float res_thres=0.5) {
            float it_sim,it_def;
            int off;
            int ix_min = floor(fpix_range_min);
            int ix_max = ceil (fpix_range_max);
            for(int k=0;k<K;k++) {
                float cur_ctf = min(ctf_info[k].x,ctf_info[k].y);

                off = k*M;
                create_ctf(ctf_fitted+off,cur_ctf);
                ctf_info[k].w = l1_norm(ctf_signal+off,ctf_fitted+off,ix_min,ix_max);

                it_def = 0;
                float w_range = def_range;
                float w_step  = def_range/10;

                do {
                    for(float delta_def=-w_range;delta_def<=w_range;delta_def+=w_step ) {
                        create_ctf(ctf_fitted+off,cur_ctf+delta_def);
                        it_sim = l1_norm(ctf_signal+off,ctf_fitted+off,ix_min,ix_max);
                        if( it_sim < ctf_info[k].w ) {
                            it_def = delta_def;
                            ctf_info[k].w = it_sim;
                        }
                    }

                    w_step = w_step/2;
                    w_range = w_range/2;
                    cur_ctf = cur_ctf+it_def;
                    it_def = 0;
                } while( w_step>def_step );

                create_ctf(ctf_fitted+off,cur_ctf);
                float ctf_dif = abs(ctf_info[k].x-ctf_info[k].y);
                ctf_info[k].x = cur_ctf + ctf_dif;
                ctf_info[k].y = cur_ctf;
                int min_ix = get_initial_ix(cur_ctf);
                calc_quality(ctf_qualit+off,min_ix,ctf_signal+off,ctf_fitted+off);
            }
        }

        void update_max_res(float2*max_res,const float4*ctf_info,float range,float res_thres=0.5) {
            int i=0;
            for(int k=0;k<K;k++) {
                int min_ix = get_initial_ix(ctf_info[k].y)+1;
                for(i=min_ix;i<M;i++) {
                    if( ctf_qualit[i+k*M] < res_thres )
                        break;
                }
                float current_res = ((float)N)*apix/((float)i);
                if( max_res[k].x < 0 || current_res < max_res[k].x ) {
                    max_res[k].x = current_res;
                    max_res[k].y = range;
                }
            }
        }

        void update_energy_shell(float2*shell_info,float4*ctf_info,float shell_value,float2 fpix_range) {
            for(int k=0;k<K;k++) {
                float energy  = 0;
                float cur_ctf = min(ctf_info[k].x,ctf_info[k].y);

                int   off = k*M;

                create_ctf(ctf_fitted+off,cur_ctf);
                energy = l1_norm(ctf_signal+off,ctf_fitted+off,floor(fpix_range.x),ceil(fpix_range.y));

                if( shell_info[k].y < 0 | energy < shell_info[k].y ) {
                    shell_info[k].x = shell_value;
                    shell_info[k].y = energy;
                }
            }
        }

        void save_svg_report(const float*ctf_avg,const float*ctf_env,float4*ctf_info,const char*out_dir,float2 fpix_range) {

            char filename[SUSAN_FILENAME_LENGTH];

            sprintf(filename,"%s/ctf_fit",out_dir);
            IO::create_dir(filename);

            float *signal_tmp = new float[M];

            for(int k=0;k<K;k++) {
                int min_ix = get_initial_ix(ctf_info[k].y);
                float nrm_max = 0;
                for(int i=0;i<M;i++) {
                    signal_tmp[i] = ctf_avg[i + k*N]+ctf_env[i + k*N];
                    if( i > min_ix ) {
                        if( ctf_env[i + k*N] > nrm_max ) {
                            nrm_max = ctf_env[i + k*N];
                        }
                    }
                }

                for(int i=0;i<M;i++) {
                    float env = 2*ctf_env[i + k*N];
                    if( i > min_ix ) {
                        env = 2*nrm_max;
                    }
                    signal_tmp[i] = signal_tmp[i]/env;
                }

                float def = (ctf_info[k].x+ctf_info[k].y)/2;
                sprintf(filename,"%s/ctf_fit/projection_%03d.svg",out_dir,k+1);
                SvgCtf report(filename,apix);
                report.create_grid(fpix_range.x,fpix_range.y,N);
                report.create_title(k+1,def);
                report.add_est(ctf_qualit+k*M,M);
                report.add_avg(signal_tmp    ,M);
                report.add_fit(ctf_fitted+k*M,M);
                report.create_legend();
            }

            delete [] signal_tmp;
        }

        void save_estimation(float4*ctf_info,float res_thres,const char*out_dir) {

            char filename[SUSAN_FILENAME_LENGTH];
            sprintf(filename,"%s/defocus.txt",out_dir);
            FILE*fp = fopen(filename,"w");
            Defocus def;
            def.ph_shft = 0;
            def.Bfactor = 0;
            def.ExpFilt = 0;

            int i=0;
            for(int k=0;k<K;k++) {
                int min_ix = get_initial_ix(ctf_info[k].y)+1;
                for(i=min_ix;i<M;i++) {
                    if( ctf_qualit[i+k*M] < res_thres )
                        break;
                }
                def.U       = ctf_info[k].x;
                def.V       = ctf_info[k].y;
                def.angle   = ctf_info[k].z*180.0/M_PI;
                def.max_res = ((float)N)*apix/((float)i);
                def.score   = ctf_info[k].w;
                IO::DefocusIO::write(fp,def);
            }

            fclose(fp);
        }

        void get_max_env_for_final_rslt(float2*p_out,const float*ctf_env,float4*ctf_info) {
            for(int k=0;k<K;k++) {
                int min_ix = get_initial_ix(ctf_info[k].y);
                float nrm_max = 0;
                for(int i=0;i<M;i++) {
                    if( i > min_ix ) {
                        if( ctf_env[i + k*N] > nrm_max ) {
                            nrm_max = ctf_env[i + k*N];
                        }
                    }
                }
                p_out[k].x = nrm_max;
                p_out[k].y = min_ix;
            }
        }

    protected:
        void create_ctf(float*p_out,float def) {
            float gamma,ctf;
            for(int i=0;i<M;i++) {
                gamma = lambda_pi*def*s2[i] - s4_lambda3_Cs_pi_2[i];
                ctf = CA*sin(gamma) + AC*cos(gamma);
                p_out[i] = ctf*ctf;
            }
        }

        float l2_norm(const float*p_sgn,const float*p_fit,int range_min,int range_max) {
            float acc = 0, tmp;
            for(int i=range_min;i<range_max;i++) {
                tmp = (p_sgn[i]-p_fit[i]);
                acc += (tmp*tmp);
            }
            acc = acc/(range_max-range_min);
            return sqrt(acc);
        }

        float l1_norm(const float*p_sgn,const float*p_fit,int range_min,int range_max) {
            float l1 = 0;
            for(int i=range_min;i<range_max;i++) {
                l1 += abs(p_sgn[i]-p_fit[i]);
            }
            return l1;
        }

        int get_initial_ix(float def) {
            int i=0;
            for(i=0;i<M;i++) {
                float gamma = lambda_pi*def*s2[i] - s4_lambda3_Cs_pi_2[i];
                if( gamma > M_PI )
                    break;
            }
            return i;
        }

        void calc_quality(float*p_out,int min_ix,const float*p_sgn,const float*p_fit) {
            float num;
            for(int i=0;i<M;i++) {
                if(i < min_ix || i == (M-1)) {
                    p_out[i] = 0.5;
                }
                else {
                    //num = min(abs(p_sgn[i-1]-p_fit[i-1]),min(abs(p_sgn[i]-p_fit[i]),abs(p_sgn[i+1]-p_fit[i+1])));
                    num = abs(p_sgn[i-1]-p_fit[i-1])+abs(p_sgn[i]-p_fit[i])+abs(p_sgn[i+1]-p_fit[i+1]);
                    num = num/3;
                    p_out[i] = min(max(float(1-num),0.0),1.0);
                }
            }
        }

        int get_res_fp(const float*p_data,int range_min,int range_max,const float res_thres) {
            int i=0;
            for(i=range_min;i<range_max;i++) {
                if( p_data[i] < res_thres )
                    break;
            }
            return i;
        }

        void get_max_idx(float&max_val,int&max_idx,const float*p_data,int range_min,int range_max) {
            max_val = p_data[range_min];
            max_idx = range_min;
            for(int i=range_min;i<range_max;i++) {
                if( max_val < p_data[i] ) {
                    max_val = p_data[i];
                    max_idx = i;
                }
            }
        }

        void get_max_idx(float&max_val,int&max_idx,const float*p_data) {
            get_max_idx(max_val,max_idx,p_data,0,M);
        }

    };

    class Result {
    public:
        float *c_rslt;
        float *c_env;
        float2 *c_min_ix;
        int N;
        int M;
        int K;
        int3 ss_c;
        int3 ss_r;
        dim3 blk;
        dim3 grd_c;
        dim3 grd_r;
        GPU::GArrSingle g_rslt;
        GPU::GArrSingle g_env;
        GPU::GArrSingle2 g_min_ix;

        Result(int m, int n, int k) {
            M = m;
            N = n;
            K = k;
            ss_c  = make_int3(m,n,k);
            ss_r  = make_int3(n,n,k);
            blk   = GPU::get_block_size_2D();
            grd_c = GPU::calc_grid_size(blk,ss_c.x,ss_c.y,ss_c.z);
            grd_r = GPU::calc_grid_size(blk,ss_r.x,ss_r.y,ss_r.z);

            g_rslt.alloc(n*n*k);
            g_env.alloc(m*k);
            g_min_ix.alloc(k);

            c_rslt = new float[N*N*K];
            c_env = new float[M*K];
            c_min_ix = new float2[K];
        }

        ~Result() {
            delete [] c_rslt;
            delete [] c_env;
            delete [] c_min_ix;
        }

        void load_env(const float*in_c_env) {
            for(int k=0;k<K;k++) {
                for(int i=0;i<M;i++) {
                    c_env[i + k*M] = in_c_env[i + k*N];
                }
            }
            cudaMemcpy( (void*)g_env.ptr, (const void*)(c_env), sizeof(float)*M*K, cudaMemcpyHostToDevice);
        }

        void gen_fitting_result(const float*g_ctf_avg,const float4*g_def_inf,const float apix,const float lambda_pi,const float lambda3_Cs_pi_2,const float ac) {
            cudaMemcpy( (void*)g_min_ix.ptr, (const void*)(c_min_ix), sizeof(float2)*K, cudaMemcpyHostToDevice);
            GpuKernelsCtf::vis_copy_data<<<grd_r,blk>>>(g_rslt.ptr,g_ctf_avg,g_env.ptr,g_min_ix.ptr,ss_r);
            GpuKernelsCtf::vis_add_ctf<<<grd_r,blk>>>(g_rslt.ptr,g_def_inf,apix,lambda_pi,lambda3_Cs_pi_2,ac,ss_r);
        }
    };

public:
    GPU::GArrSingle2  ss_lin_mask;
    GPU::GArrSingle4  g_def_inf;

    float2 *lin_mask;
    float2 *max_res_log;
    float4 *c_def_inf;
    float4 *c_def_rslt;
    float  *p_rad_avg;
    float  *p_rad_env;
    float  *p_ps_lin_avg;
    float  *p_final_rslt;

    float  apix;
    float  apix_scaled;
    float2 fpix_range;
    float2 def_lin_range;
    float  ref_step;
    float  ref_range;
    float  lambda;
    float  lambda_pi;
    float  lambda3_Cs_pi_2;
    float  ix2def;
    float  tlt_fpix;
    float  linearization_scale;

    float  AC;
    float  CA;

    float N;
    float M;
    float K;

    int   verbose;
    float res_thres;
    float bfac_max;

    char filename[SUSAN_FILENAME_LENGTH];

    CtfLinearizer(int gpu_ix, int n, int k) {

        N = n;
        M = n/2 + 1;
        K = k;

        GPU::set_device(gpu_ix);

        ss_lin_mask.alloc(K);
        g_def_inf.alloc(K);


        lin_mask  = new float2[int(K)];
        p_rad_avg = new float[int(N*K)];
        p_rad_env = new float[int(N*K)];
        c_def_inf = new float4[int(K)];
        c_def_rslt = new float4[int(K)];
        max_res_log = new float2[int(K)];
        p_ps_lin_avg = new float[int(M*N)];
        p_final_rslt = new float[int(N*N*K)];
    }

    ~CtfLinearizer() {
        delete [] lin_mask;
        delete [] p_rad_avg;
        delete [] p_rad_env;
        delete [] c_def_inf;
        delete [] c_def_rslt;
        delete [] max_res_log;
        delete [] p_ps_lin_avg;
        delete [] p_final_rslt;
    }
	
    void load_info(ArgsCTF::Info*info,Tomogram*p_tomo) {

        verbose   = info->verbose;
        res_thres = info->res_thres;
        bfac_max  = info->bfac_max;

        lambda = Math::get_lambda(p_tomo->KV);
        lambda_pi = lambda*M_PI;
        lambda3_Cs_pi_2 = lambda*lambda*lambda*(p_tomo->CS*1e7)*M_PI/2;
        apix = p_tomo->pix_size*pow(2.0,info->binning);

        AC = p_tomo->AC;
        CA = sqrt(1.0-AC*AC);

        ref_step = info->ref_step;
        ref_range = info->ref_range;

        fpix_range.x = N*apix/info->res_min;
        fpix_range.y = N*apix/info->res_max;
        fpix_range.x = max(fpix_range.x-5,5.0);
        fpix_range.y = min(fpix_range.y+5,M-7.0);

        linearization_scale = (2.3*fpix_range.y/N);

        apix_scaled = apix/linearization_scale;
        ix2def = 2*apix_scaled*apix_scaled/lambda;

        def_lin_range.x = info->def_min/ix2def;
        def_lin_range.y = info->def_max/ix2def;
        def_lin_range.x = max(def_lin_range.x-2,3.0);
        def_lin_range.y = min(def_lin_range.y+2,M-4.0);

        tlt_fpix = ceilf(2+info->tlt_range/ix2def);
    }
	
    void process(const char*out_dir,float*input,Tomogram*p_tomo) {

        float max_lin_r_px = calc_initial_range_max(def_lin_range.x);

        for(int k=0;k<K;k++) {
            max_res_log[k].x = -1; /// max res
            max_res_log[k].y =  0; /// shells

            c_def_rslt[k].w = -1;
        }

        Normal ctf_normal(M,N,K);
        Linear ctf_linear(M,N,K);
        FitEllipse fit_ellipe(90);
        FitCtf fit_ctf(M,N,K,apix,lambda_pi,lambda3_Cs_pi_2,AC);
        RadialAvgr rad_avgr(M,N,K,ix2def,lambda_pi,apix);
        Result results(M,N,K);

        ctf_normal.load_rmv_bg_msk(input,fpix_range);
        save_gpu_mrc(input,ctf_normal.ss_fg_masked.ptr,M,N,K,out_dir,"ctf_normalized.mrc",1);

        ctf_linear.load_linearize(ctf_normal.ss_fg_masked.ptr,linearization_scale);
        save_gpu_mrc(input,ctf_linear.r_linear_in.ptr,M,N,K,out_dir,"ctf_linearized.mrc",2);

        set_lin_range(10.0,max_lin_r_px);
        ctf_linear.mask_and_power_spectrum(ss_lin_mask.ptr,def_lin_range);
        save_gpu_mrc(input,ctf_linear.ss_pow_spct.ptr,M,N,K,out_dir,"ctf_ps_lin_raw.mrc",2);

        float3 avg_def;
        ctf_linear.sum_z();
        save_gpu_mrc(p_ps_lin_avg,ctf_linear.ss_ps_sum_z.ptr,M,N,1,out_dir,"ctf_ps_lin_avg.mrc",1);
        fit_ellipe.fit(avg_def.x,avg_def.y,avg_def.z,p_ps_lin_avg,M,N);
        printf(" Average defocus (Angstroms): U=%9.2f, V=%9.2f, angle=%6.1f.\n",
               avg_def.x*ix2def,avg_def.y*ix2def,avg_def.z*180.0/M_PI);

        if( verbose >= 3 )
            fit_ellipe.save_points("plane_pts.txt",out_dir);

        ctf_linear.load_def_range(avg_def.x,avg_def.y,avg_def.z,tlt_fpix,p_tomo);

        float def_avg        = (avg_def.x+avg_def.y)/2;
        float min_lin_r_px   = calc_range_min(def_avg);

        for(int i=0;i<6;i++) {
            char tmp[1024];

            max_lin_r_px = calc_range_max(i,def_avg);
            set_lin_range(min_lin_r_px,max_lin_r_px);
            ctf_linear.mask_and_power_spectrum(ss_lin_mask.ptr,def_lin_range);
            ctf_linear.apply_def_range();

            sprintf(tmp,"ctf_ps_lin_%d.mrc",i);
            save_gpu_mrc(input,ctf_linear.ss_pow_spct.ptr,M,N,K,out_dir,tmp,3);

            for(int k=0;k<K;k++) {
                fit_ellipe.fit(c_def_inf[k].x,c_def_inf[k].y,c_def_inf[k].z,input+k*int(M*N),M,N);
            }
            update_def_info();
            rad_avgr.load_and_radial_normalize(ctf_normal.ss_foreground.ptr,g_def_inf.ptr);
            rad_avgr.calc_rad_average(p_rad_avg);
            rad_avgr.calc_env(p_rad_env,p_rad_avg);

            sprintf(tmp,"ctf_rad_norm_%d.mrc",i);
            save_gpu_mrc(input,rad_avgr.rad_normal.ptr,M,N,K,out_dir,tmp,3);


            for(int k=0;k<K;k++) {
                c_def_inf[k].x *= ix2def;
                c_def_inf[k].y *= ix2def;
            }
            fit_ctf.load_ctf_avg(p_rad_avg,p_rad_env);
            fit_ctf.estimate(c_def_inf,ref_range,ref_step,fpix_range.x,calc_fpix_max(i,def_avg),0.5);
            fit_ctf.update_max_res(max_res_log,c_def_inf,max_lin_r_px,res_thres);

            sprintf(tmp,"ctf_rad_avg_raw_%d.mrc",i);
            save_cpu_mrc(p_rad_avg,N,K,1,out_dir,tmp,3);

            sprintf(tmp,"ctf_rad_avg_est_%d.mrc",i);
            save_cpu_mrc(fit_ctf.ctf_fitted,M,K,1,out_dir,tmp,3);
        }

        set_lin_range(min_lin_r_px,max_res_log);
        ctf_linear.mask_and_power_spectrum(ss_lin_mask.ptr,def_lin_range);
        ctf_linear.apply_def_range();
        save_gpu_mrc(input,ctf_linear.ss_pow_spct.ptr,M,N,K,out_dir,"ctf_ps_lin.mrc",1);

        for(int k=0;k<K;k++) {
            fit_ellipe.fit(c_def_inf[k].x,c_def_inf[k].y,c_def_inf[k].z,input+k*int(M*N),M,N);
        }

        update_def_info();
        for(int k=0;k<K;k++) {
            c_def_inf[k].x *= ix2def;
            c_def_inf[k].y *= ix2def;
        }

        rad_avgr.load_and_radial_normalize(ctf_normal.ss_foreground.ptr,g_def_inf.ptr);
        rad_avgr.calc_rad_average(p_rad_avg);
        rad_avgr.calc_env(p_rad_env,p_rad_avg);

        fit_ctf.load_ctf_avg(p_rad_avg,p_rad_env);
        fit_ctf.estimate(c_def_inf,ref_range,ref_step,fpix_range.x,fpix_range.y,res_thres);
        fit_ctf.save_estimation(c_def_inf,res_thres,out_dir);
        if( verbose >= 2 )
            fit_ctf.save_svg_report(p_rad_avg,p_rad_env,c_def_inf,out_dir,fpix_range);

        save_cpu_mrc(p_rad_avg,N,K,1,out_dir,"ctf_rad_avg_raw.mrc",2);
        save_cpu_mrc(p_rad_env,N,K,1,out_dir,"ctf_rad_avg_env.mrc",2);
        save_cpu_mrc(fit_ctf.ctf_signal,M,K,1,out_dir,"ctf_rad_avg_nrm.mrc",2);
        save_cpu_mrc(fit_ctf.ctf_fitted,M,K,1,out_dir,"ctf_rad_avg_est.mrc",2);
        save_cpu_mrc(fit_ctf.ctf_qualit,M,K,1,out_dir,"ctf_rad_avg_qly.mrc",2);

        update_def_info();
        fit_ctf.get_max_env_for_final_rslt(results.c_min_ix,p_rad_env,c_def_inf);
        results.load_env(p_rad_env);
        results.gen_fitting_result(ctf_normal.ss_foreground.ptr,g_def_inf.ptr,apix,lambda_pi,lambda3_Cs_pi_2,AC);
        save_gpu_mrc(p_final_rslt,results.g_rslt.ptr,N,N,K,out_dir,"ctf_fitting_result.mrc",0);

    }
	
protected:
    void set_lin_range(const float min_fp,const float max_fp) {
        for(int k=0;k<K;k++) {
            lin_mask[k].x = min_fp;
            lin_mask[k].y = max_fp;
        }
        cudaMemcpy( (void*)ss_lin_mask.ptr, (const void*)(lin_mask), sizeof(float2)*K, cudaMemcpyHostToDevice);
    }

    void set_lin_range(const float min_fp,const float2*factor_per_proj) {
        for(int k=0;k<K;k++) {
            lin_mask[k].x = min_fp;
            lin_mask[k].y = factor_per_proj[k].y;
        }
        cudaMemcpy( (void*)ss_lin_mask.ptr, (const void*)(lin_mask), sizeof(float2)*K, cudaMemcpyHostToDevice);
    }

    void download(single*p_cpu,const single*p_gpu,const int m,const int n,const int k) {
        cudaMemcpy((void*)p_cpu, (const void*)p_gpu, sizeof(float)*m*n*k, cudaMemcpyDeviceToHost);
    }

    void save_gpu_mrc(single*p_cpu,const single*p_gpu,const int m,const int n,const int k,const char*out_dir,const char*name,const int req_verb) {
        cudaMemcpy((void*)p_cpu, (const void*)p_gpu, sizeof(float)*m*n*k, cudaMemcpyDeviceToHost);
        if( verbose >= req_verb ) {
            sprintf(filename,"%s/%s",out_dir,name);
            Mrc::write(p_cpu,m,n,k,filename);
        }
    }

    void save_cpu_mrc(single*p_cpu,const int m,const int n,const int k,const char*out_dir,const char*name,const int req_verb) {
        if( verbose >= req_verb ) {
            sprintf(filename,"%s/%s",out_dir,name);
            Mrc::write(p_cpu,m,n,k,filename);
        }
    }

    void update_def_info() {
        cudaMemcpy( (void*)g_def_inf.ptr, (const void*)(c_def_inf), sizeof(float4)*K, cudaMemcpyHostToDevice);
    }

    float calc_initial_range_max(float max_range) {
        return min(2.5*max_range,M-10);
    }

    float calc_range_min(float cur_def) {
        return N/(4*cur_def);
    }

    float calc_range_max(float i,float cur_def) {
        return min((i+1)*N/cur_def,M-10);
    }

    float calc_fpix_max(float i,float cur_def) {
        float tmp = 2*(i+1);
        tmp = tmp/(lambda*cur_def);
        return N*apix*sqrt(tmp);
    }
};

#endif /// CTF_LINEARIZER_H


