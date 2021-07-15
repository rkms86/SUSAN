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

            create_filter();
        }

        void load_rmv_bg_msk(const float*cpu_input,float2 fpix_range) {
            /// use ss_fg_masked as an upload buffer:
            cudaMemcpy( (void*)ss_fg_masked.ptr, (const void*)(cpu_input), sizeof(float)*numel, cudaMemcpyHostToDevice);
            GpuKernelsCtf::rmv_bg<<<grd_c,blk>>>(ss_foreground.ptr,ss_fg_masked.ptr,ss_filter.ptr,n_filter,ss_c);
            GpuKernels::conv_gaussian<<<grd_c,blk>>>(ss_fg_masked.ptr,ss_foreground.ptr,ss_c);
            cudaMemcpy( (void*)ss_foreground.ptr, (const void*)(ss_fg_masked.ptr), sizeof(float)*numel, cudaMemcpyDeviceToDevice);
            GpuKernelsCtf::keep_fpix_range<<<grd_c,blk>>>(ss_fg_masked.ptr,fpix_range,ss_c);
        }

    protected:
        GPU::GArrSingle3 ss_filter;
        uint32           n_filter;

        void create_filter() {
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
            GpuKernels::load_surf<<<grd_c,blk>>>(ss_lin.surface,gpu_input,ss_c);
            GpuKernelsCtf::ctf_linearize<<<grd_c,blk>>>(r_linear_in.ptr,ss_lin.texture,linearization_scale,ss_c);
            GpuKernels::load_surf<<<grd_c,blk>>>(ss_lin.surface,r_linear_in.ptr,ss_c);
            GpuKernelsCtf::tangential_blur<<<grd_c,blk>>>(r_linear_in.ptr,ss_lin.texture,ss_c);
            GpuKernels::expand_ps_hermitian<<<grd_r,blk>>>(r_lin_expnd.ptr,r_linear_in.ptr,ss_r);
        }

        void mask_and_power_spectrum(const float2*gpu_fp_mask,const float2 def_lin_range) {
            GpuKernels::apply_circular_mask<<<grd_r,blk>>>(r_lin_mskd.ptr,r_lin_expnd.ptr,gpu_fp_mask,ss_r);
            GpuKernels::fftshift2D<<<grd_r,blk>>>(r_lin_mskd.ptr,ss_r);
            fft2.exec(c_linear.ptr,r_lin_mskd.ptr);
            GpuKernels::fftshift2D<<<grd_c,blk>>>(c_linear.ptr,ss_c);
            GpuKernels::load_surf_real<<<grd_c,blk>>>(ss_lin.surface,c_linear.ptr,ss_c);
            GpuKernelsCtf::tangential_blur<<<grd_c,blk>>>(ss_pow_spct.ptr,ss_lin.texture,ss_c);
            GpuKernels::load_surf<<<grd_c,blk>>>(ss_lin.surface,ss_pow_spct.ptr,ss_c);
            GpuKernelsCtf::radial_edge_detect<<<grd_c,blk>>>(ss_pow_spct.ptr,ss_lin.texture,ss_c);
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
                c_def[k].w = 6.0*(1-abs(cos(euZYZ(1)))) + tlt_fpix;
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

            Math::fit_ellipsoid(def_u,def_v,def_ang,points);
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

        void estimate(float4*ctf_info,float def_range,float def_step,float2 fpix_range) {
            float it_sim,it_def;
            int off;
            int ix_min = floor(fpix_range.x);
            int ix_max = ceil (fpix_range.y);
            for(int k=0;k<K;k++) {
                float cur_ctf = min(ctf_info[k].x,ctf_info[k].y);
                off = k*M;
                create_ctf(ctf_fitted+off,cur_ctf);
                it_def = 0;
                ctf_info[k].w = l1_norm(ctf_signal+off,ctf_fitted+off,ix_min,ix_max);

                float w_range = def_range;
                float w_step  = def_range/10;

                do {
                    it_def = 0;
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
                } while( w_step>def_step );

                create_ctf(ctf_fitted+off,cur_ctf);
                float ctf_dif = abs(ctf_info[k].x-ctf_info[k].y);
                ctf_info[k].x = cur_ctf + ctf_dif;
                ctf_info[k].y = cur_ctf;
                int min_ix = get_initial_ix(cur_ctf);
                calc_quality(ctf_qualit+off,min_ix,ctf_signal+off,ctf_fitted+off);
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
    float2 *rad_shell_energy;
    float4 *c_def_inf;
    float4 *c_def_rslt;
    float  *p_rad_avg;
    float  *p_rad_env;
    float  *p_ps_lin_avg;
    float  *p_final_rslt;

        /*int3  ss_c;
	int3  ss_r;
        int3  ss_2;
	dim3  blk;
	dim3  grd_c;
	dim3  grd_r;
        dim3  grd_2;
	GPU::GArrSingle   ss_input;
	GPU::GArrSingle   ss_sum_z;
	GPU::GArrSingle   ss_data_c;
	GPU::GArrSingle   ss_data_r;
        GPU::GArrSingle   ss_normal_c;
        GPU::GArrSingle   ss_linear_r;
	GPU::GArrSingle   ss_vis;
	GPU::GArrSingle2  ss_data_R;
        //GPU::GArrSingle3  ss_filter;
        GPU::GArrSingle   rad_avg;
        GPU::GArrSingle   rad_wgt;
	GPU::GArrSingle2  rad_fourier;
	GPU::GArrSingle2  rad_hilbert;
	GpuFFT::FFT2D     fft2;
	GPU::GTex2DSingle ss_lin;
	
        //float  *p_rad_avg;
        //float  *p_rad_wgt;
        float  *p_rad_nrm;
        float  *p_ctf_fit;
        float  *p_ctf_res;*/
	
    uint32 n_filter;
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
        p_ps_lin_avg = new float[int(M*N)];
        p_final_rslt = new float[int(N*N*K)];
        rad_shell_energy = new float2[int(K)];

            /*ss_vis.alloc(N*N*K);
            ss_input.alloc(M*N*K);
            ss_sum_z.alloc(M*N);
            ss_data_c.alloc(M*N*K);
            ss_data_r.alloc(N*N*K);
            ss_data_R.alloc(M*N*K);
            ss_normal_c.alloc(M*N*K);
            ss_linear_r.alloc(N*N*K);

            g_def_inf.alloc(K);

            rad_avg.alloc(M*K);
            rad_wgt.alloc(M*K);
            rad_fourier.alloc(M*K);
            rad_hilbert.alloc(M*K);

            ss_lin.alloc(M,N,K);
            fft2.alloc(M,N,K);

            //create_filter();

            p_ps_lin_avg = new float[int(M*N)];
            c_def_inf = new float4[int(K)];
            p_rad_nrm = new float[int(M*K)];
            p_ctf_fit = new float[int(M*K)];
            p_ctf_res = new float[int(M*K)];
            */
    }

    ~CtfLinearizer() {
        delete [] lin_mask;
        delete [] p_rad_avg;
        delete [] p_rad_env;
        delete [] c_def_inf;
        delete [] c_def_rslt;
        delete [] p_ps_lin_avg;
        delete [] rad_shell_energy;
        delete [] p_final_rslt;

        /*
        delete [] c_def_inf;
        delete [] p_rad_nrm;
        delete [] p_ctf_fit;
        delete [] p_ctf_res;
        */
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

        float min_lin_r_px = 3.0;
        float delta_lin_r_px = M-min_lin_r_px;

        for(int k=0;k<K;k++) {
            rad_shell_energy[k].x =  0;
            rad_shell_energy[k].y = -1;

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

        ctf_linear.load_linearize(ctf_normal.ss_foreground.ptr,linearization_scale);
        save_gpu_mrc(input,ctf_linear.r_linear_in.ptr,M,N,K,out_dir,"ctf_linearized.mrc",2);

        set_lin_range(min_lin_r_px,min_lin_r_px+0.5*delta_lin_r_px);
        ctf_linear.mask_and_power_spectrum(ss_lin_mask.ptr,def_lin_range);
        save_gpu_mrc(input,ctf_linear.ss_pow_spct.ptr,M,N,K,out_dir,"ctf_ps_lin_raw.mrc",2);

        float3 avg_def;
        ctf_linear.sum_z();
        save_gpu_mrc(p_ps_lin_avg,ctf_linear.ss_ps_sum_z.ptr,M,N,1,out_dir,"ctf_ps_lin_avg.mrc",1);
        fit_ellipe.fit(avg_def.x,avg_def.y,avg_def.z,p_ps_lin_avg,M,N);
        printf("          - Average defocus (Angstroms): U=%.2f, V=%.2f, angle=%.1f.\n",
               avg_def.x*ix2def,avg_def.y*ix2def,avg_def.z*180.0/M_PI);

        ctf_linear.load_def_range(avg_def.x,avg_def.y,avg_def.z,tlt_fpix,p_tomo);

        for(int i=0;i<5;i++) {
            char tmp[1024];

            // (1:5)/8 = ~(0.125:0.125:0.625)
            // (1:5)/9 = ~(0.111:0.111:0.556)
            float factor = float(i+1)/9;
            set_lin_range(min_lin_r_px,min_lin_r_px+factor*delta_lin_r_px);
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
            fit_ctf.estimate(c_def_inf,ref_range,ref_step,fpix_range);

            for(int k=0;k<K;k++) {
                if( (c_def_rslt[k].w < 0) || (c_def_inf[k].w < c_def_rslt[k].w) ) {
                    c_def_rslt[k].x = c_def_inf[k].x;
                    c_def_rslt[k].y = c_def_inf[k].y;
                    c_def_rslt[k].z = c_def_inf[k].z;
                    c_def_rslt[k].w = c_def_inf[k].w;
                    rad_shell_energy[k].x = factor;
                }
            }
        }

        set_lin_range(min_lin_r_px,delta_lin_r_px,rad_shell_energy);
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
        fit_ctf.estimate(c_def_inf,ref_range,ref_step,fpix_range);
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

            //fit_ctf.load_and_radial_normalize(ctf_normal.ss_foreground.ptr,g_def_inf.ptr);

/*		rad_avg.clear();
		rad_wgt.clear();
		
                // ss_input <- ctf_normalized;
                // remove_background(ss_input);
                // keep_data_in_resolution_range(ss_input);
                cudaMemcpy( (void*)ss_input.ptr, (const void*)(input), sizeof(float)*ss_c.x*ss_c.y*ss_c.z, cudaMemcpyHostToDevice);
                GpuKernelsCtf::rmv_bg<<<grd_c,blk>>>(ss_normal_c.ptr,ss_input.ptr,ss_filter.ptr,n_filter,ss_c);
                cudaMemcpy( (void*)ss_data_c.ptr, (const void*)(ss_normal_c.ptr), sizeof(float)*ss_c.x*ss_c.y*ss_c.z, cudaMemcpyDeviceToDevice);
                GpuKernelsCtf::keep_fpix_range<<<grd_c,blk>>>(ss_data_c.ptr,fpix_range,ss_c);

                //save_gpu_mrc(input,ss_normal_c.ptr,ss_c.x,ss_c.y,ss_c.z,out_dir,"ctf_normalized.mrc",1);
		
                // surface <- ss_input;
                // ss_data <- linearize(surface);
                // surface <- ss_data;
                // ss_data <- blur_radially(surface);
                GpuKernels::load_surf<<<grd_c,blk>>>(ss_lin.surface,ss_data_c.ptr,ss_c);
		GpuKernelsCtf::ctf_linearize<<<grd_c,blk>>>(ss_data_c.ptr,ss_lin.texture,linearization_scale,ss_c);
		GpuKernels::load_surf<<<grd_c,blk>>>(ss_lin.surface,ss_data_c.ptr,ss_c);
                GpuKernelsCtf::tangential_blur<<<grd_c,blk>>>(ss_data_c.ptr,ss_lin.texture,ss_c);
		
                save_gpu_mrc(input,ss_data_c.ptr,ss_c.x,ss_c.y,ss_c.z,out_dir,"ctf_linearized.mrc",2);

                // ss_linear_r <- expand(ss_data);
                // ss_data_r <- mask(ss_linear_r,0.5);
                // ss_lin = max(fft2(ss_data_r),0);
                // ss_data_c = mask( highpass( blur( ss_lin ) ) );
                set_max_r(0.5);
                GpuKernels::expand_ps_hermitian<<<grd_r,blk>>>(ss_linear_r.ptr,ss_data_c.ptr,ss_r);
                GpuKernels::apply_circular_mask<<<grd_r,blk>>>(ss_data_r.ptr,ss_linear_r.ptr,g_def_inf.ptr,ss_r);
		GpuKernels::fftshift2D<<<grd_r,blk>>>(ss_data_r.ptr,ss_r);
		fft2.exec(ss_data_R.ptr,ss_data_r.ptr);
		GpuKernels::fftshift2D<<<grd_c,blk>>>(ss_data_R.ptr,ss_c);
                GpuKernels::load_surf_real_positive<<<grd_c,blk>>>(ss_lin.surface,ss_data_R.ptr,ss_c);
		GpuKernelsCtf::tangential_blur<<<grd_c,blk>>>(ss_data_c.ptr,ss_lin.texture,ss_c);
		GpuKernels::load_surf<<<grd_c,blk>>>(ss_lin.surface,ss_data_c.ptr,ss_c);
		GpuKernelsCtf::radial_highpass<<<grd_c,blk>>>(ss_data_c.ptr,ss_lin.texture,ss_c);
		GpuKernelsCtf::keep_fpix_range<<<grd_c,blk>>>(ss_data_c.ptr,def_lin_range,ss_c);
		
		save_gpu_mrc(input,ss_data_c.ptr,ss_c.x,ss_c.y,ss_c.z,out_dir,"ctf_ps_lin_raw.mrc",1);
		
                // ss_sum = sum( ss_data_c, Z );
		GpuKernelsCtf::sum_along_z<<<grd_c,blk>>>(ss_sum_z.ptr,ss_data_c.ptr,ss_c);
		
                // Estimate defocus of the average ctf
		float3 avg_def;
		save_gpu_mrc(p_ps_lin_avg,ss_sum_z.ptr,ss_c.x,ss_c.y,1,out_dir,"ctf_ps_lin_avg.mrc",1);
		get_avg_defocus(avg_def,p_ps_lin_avg);
                printf("          - Average defocus (Angstroms): U=%.2f, V=%.2f, angle=%.1f (Linear fourier pixel size: %.1f)\n",avg_def.x*ix2def,avg_def.y*ix2def,avg_def.z*180.0/M_PI,ix2def);
		
                set_def_info(avg_def,p_tomo);
                //GpuKernelsCtf::mask_ellipsoid<<<grd_c,blk>>>(ss_data_c.ptr,g_def_inf.ptr,ss_c);
		save_gpu_mrc(input,ss_data_c.ptr,ss_c.x,ss_c.y,ss_c.z,out_dir,"ctf_ps_lin.mrc",2);
		

                for(int i=0;i<5;i++) {
                    char tmp[1024];
                    sprintf(tmp,"ctf_ps_lin_%d.mrc",i);
                    set_max_r(float(i)/5);
                    GpuKernels::apply_circular_mask<<<grd_r,blk>>>(ss_data_r.ptr,ss_linear_r.ptr,g_def_inf.ptr,ss_r);
                    GpuKernels::fftshift2D<<<grd_r,blk>>>(ss_data_r.ptr,ss_r);
                    fft2.exec(ss_data_R.ptr,ss_data_r.ptr);
                    GpuKernels::fftshift2D<<<grd_c,blk>>>(ss_data_R.ptr,ss_c);
                    GpuKernels::load_surf_real_positive<<<grd_c,blk>>>(ss_lin.surface,ss_data_R.ptr,ss_c);
                    GpuKernelsCtf::tangential_blur<<<grd_c,blk>>>(ss_data_c.ptr,ss_lin.texture,ss_c);
                    //GpuKernels::load_surf<<<grd_c,blk>>>(ss_lin.surface,ss_data_c.ptr,ss_c);
                    //GpuKernelsCtf::radial_highpass<<<grd_c,blk>>>(ss_data_c.ptr,ss_lin.texture,ss_c);
                    GpuKernelsCtf::keep_fpix_range<<<grd_c,blk>>>(ss_data_c.ptr,def_lin_range,ss_c);
                    save_gpu_mrc(input,ss_data_c.ptr,ss_c.x,ss_c.y,ss_c.z,out_dir,tmp,2);
                    get_initial_defocus(avg_def,input);
                    GpuKernels::load_surf<<<grd_c,blk>>>(ss_lin.surface,ss_normal_c.ptr,ss_c);
                    GpuKernelsCtf::ctf_radial_normalize<<<grd_c,blk>>>(ss_data_c.ptr,ss_lin.texture,g_def_inf.ptr,ix2def,M_PI*lambda,apix,ss_c);
                    sprintf(tmp,"ctf_rad_norm_%d.mrc",i);
                    save_gpu_mrc(input,ss_data_c.ptr,ss_c.x,ss_c.y,ss_c.z,out_dir,tmp,2);

                }

                int n_avg_f = 8;
		float2 tmp_range;
		tmp_range.x = n_avg_f;
		tmp_range.y = M-n_avg_f-1;
		float lambda_def = ix2def*lambda*(avg_def.x+avg_def.y)/2;
		GpuKernelsCtf::rmv_bg<<<grd_c,blk>>>(ss_data_c.ptr,ss_input.ptr,ss_filter.ptr,n_filter,ss_c);
		GpuKernelsCtf::keep_fpix_range<<<grd_c,blk>>>(ss_data_c.ptr,tmp_range,ss_c);
		GpuKernels::load_surf<<<grd_c,blk>>>(ss_lin.surface,ss_data_c.ptr,ss_c);
		GpuKernelsCtf::ctf_radial_normalize<<<grd_c,blk>>>(ss_data_c.ptr,ss_lin.texture,g_def_inf.ptr,ix2def,M_PI*lambda,apix,ss_c);
		save_gpu_mrc(input,ss_data_c.ptr,ss_c.x,ss_c.y,ss_c.z,out_dir,"ctf_rad_norm.mrc",2);
		GpuKernels::radial_ps_avg<<<grd_c,blk>>>(rad_avg.ptr,rad_wgt.ptr,ss_data_c.ptr,ss_c);
		GpuKernels::divide<<<grd_2,blk>>>(rad_avg.ptr,rad_wgt.ptr,ss_2);
		GpuKernelsCtf::rmv_bg<<<grd_2,blk>>>(rad_wgt.ptr,rad_avg.ptr,lambda_def,ss_2);
		cudaMemcpy((void*)p_rad_avg, (const void*)rad_wgt.ptr, sizeof(float)*ss_2.x*ss_2.y*ss_2.z, cudaMemcpyDeviceToHost);
                calculate_hilbert(rad_hilbert.ptr,rad_wgt.ptr,ss_2.y);
                GpuKernels::load_abs<<<grd_2,blk>>>(rad_wgt.ptr,rad_hilbert.ptr,ss_2);
                cudaMemcpy((void*)p_rad_wgt, (const void*)rad_wgt.ptr, sizeof(float)*ss_2.x*ss_2.y*ss_2.z, cudaMemcpyDeviceToHost);
		adjust_radial_averages(ss_2.y,out_dir);
		generate_ctf(ss_2.y);
		generate_phase_dif(ss_2.y);
		save_defocus(ss_2.y,out_dir);
		
		save_svg_report(ss_2.y,out_dir,2);
		
		cudaMemcpy((void*)p_rad_wgt, (const void*)rad_wgt.ptr, sizeof(float)*ss_2.x*ss_2.y*ss_2.z, cudaMemcpyDeviceToHost);
                //GpuKernelsCtf::rmv_bg<<<grd_c,blk>>>(ss_data_c.ptr,ss_input.ptr,ss_filter.ptr,n_filter,ss_c);
                GpuKernelsCtf::tangential_blur<<<grd_c,blk>>>(ss_data_c.ptr,ss_lin.texture,ss_c);
                GpuKernelsCtf::keep_fpix_range<<<grd_c,blk>>>(ss_data_c.ptr,tmp_range,ss_c);
                save_fitted_defocus(ss_2.y,out_dir);*/
    }
	
protected:
    void set_lin_range(const float min_fp,const float max_fp) {
        for(int k=0;k<K;k++) {
            lin_mask[k].x = min_fp;
            lin_mask[k].y = max_fp;
        }
        cudaMemcpy( (void*)ss_lin_mask.ptr, (const void*)(lin_mask), sizeof(float2)*K, cudaMemcpyHostToDevice);
    }

    void set_lin_range(const float min_fp,const float delta_fp, const float2*factor_per_proj) {
        for(int k=0;k<K;k++) {
            lin_mask[k].x = min_fp;
            lin_mask[k].y = min_fp + factor_per_proj[k].x*delta_fp;
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


/*
        void create_filter() {
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

        void get_avg_defocus(float3&defocus,const single*p_data) {

            int N_points = 90;
            Eigen::MatrixXf points = Eigen::MatrixXf::Zero(2,N_points);

            float x_c = 0;
            float y_c = N/2;

            for(int n_p=0;n_p<N_points;n_p++) {

                float ang = n_p;
                ang *= (M_PI/N_points);
                ang = ang - (M_PI/2);

                float x_v = cos(ang);
                float y_v = sin(ang);

                float max_val=0,max_x=0,max_y=0;

                for(float r=0;r<M-3;r+=1.0) {
                    float x = x_c + r*x_v;
                    float y = y_c + r*y_v;
                    int xx = (int)round(x);
                    int yy = (int)round(y);
                    float val = p_data[ (int)xx + yy*(int)M ];
                    if( val > max_val ) {
                        max_val = val;
                        max_x = x;
                        max_y = y-y_c;
                    }
                }
                points(0,n_p)=max_x;
                points(1,n_p)=max_y;
            }

            Math::fit_ellipsoid(defocus.x,defocus.y,defocus.z,points);
		
	}
	
        void set_max_r(const float max_r) {
            for(int k=0;k<ss_c.z;k++) {
                c_def_inf[k].w = max_r*float(M);
            }
            cudaMemcpy( (void*)g_def_inf.ptr, (const void*)(c_def_inf), sizeof(float4)*ss_c.z, cudaMemcpyHostToDevice);
        }

	void set_def_info(const float3&defocus,Tomogram*p_tomo) {
                float L = abs(defocus.x-defocus.y)/2;
		for(int k=0;k<ss_c.z;k++) {
			V3f euZYZ;
			Math::Rmat_eZYZ(euZYZ,p_tomo->R[k]);
			c_def_inf[k].x = defocus.x;
			c_def_inf[k].y = defocus.y;
			c_def_inf[k].z = defocus.z;
			c_def_inf[k].w = L*(1-abs(cos(euZYZ(1)))) + tlt_fpix;
		}
		
		cudaMemcpy( (void*)g_def_inf.ptr, (const void*)(c_def_inf), sizeof(float4)*ss_c.z, cudaMemcpyHostToDevice);
	}
	
	void get_initial_defocus(const float3&defocus,const single*p_data) {
		float avg_def = (defocus.x+defocus.y)/2;
		float ang_res = atan2(1,avg_def);
		int num_pts = round(M_PI/ang_res) + 1;
		ang_res = M_PI/(num_pts-1);
		Eigen::MatrixXf pts = Eigen::MatrixXf::Zero(2,num_pts);
		
		for(int k=0;k<ss_c.z;k++) {
			int off = (int)(k*N*M);
			extract_radial_peaks(pts,p_data + off,ang_res,avg_def-c_def_inf[k].w,avg_def+c_def_inf[k].w);
			Math::fit_ellipsoid(c_def_inf[k].x,c_def_inf[k].y,c_def_inf[k].z,pts);
		}
		cudaMemcpy( (void*)g_def_inf.ptr, (const void*)(c_def_inf), sizeof(float4)*ss_c.z, cudaMemcpyHostToDevice);
	}
	
	void extract_radial_peaks(Eigen::MatrixXf&pts,const single*p_data,const float ang_res,const float min_r,const float max_r) {
		int num_pts = pts.cols();
		float Nh = N/2;
		for(int i=0;i<num_pts;i++) {
			float x = cos(ang_res*i);
			float y = sin(ang_res*i);
			float m_val = 0;
			for(float r = min_r; r<=max_r; r+=1.0f ) {
				float xw = x*r;
				float yw = y*r;
				if(xw<0) {
					xw = -xw;
					yw = -yw;
				}

				float c_val = 0;
				int x_val = (int)floorf(xw);
				int y_val = (int)floorf(yw+Nh);
				float x_wgt = xw - x_val;
				float y_wgt = yw+Nh - y_val;
				c_val += (1-x_wgt)*(1-y_wgt)*p_data[ x_val   + int(M)*(y_val  ) ];
				c_val += (  x_wgt)*(1-y_wgt)*p_data[ x_val+1 + int(M)*(y_val  ) ];
				c_val += (1-x_wgt)*(  y_wgt)*p_data[ x_val   + int(M)*(y_val+1) ];
				c_val += (  x_wgt)*(  y_wgt)*p_data[ x_val+1 + int(M)*(y_val+1) ];
				if( c_val > m_val ) {
					m_val = c_val;
					pts(0,i)=xw;
					pts(1,i)=yw;
				}
			}
		}
	}
	
        void calculate_hilbert(float2*p_out,single*p_in,const int k) {
		GpuFFT::FFT1D_full  fft1fwd;
		GpuFFT::IFFT1D_full fft1inv;
                fft1fwd.alloc(M,k);
                fft1inv.alloc(M,k);
                fft1fwd.exec(rad_fourier.ptr,p_in);
                GpuKernelsCtf::prepare_hilbert<<<grd_2,blk>>>(rad_fourier.ptr,(int)roundf(N/4),ss_2);
                fft1inv.exec(p_out,rad_fourier.ptr);
	}
	
	void adjust_radial_averages(const int k,const char*out_dir) {
		float*p_nrm;
		float*p_avg;
		float*p_wgt;
		
		if( verbose >= 2 ) {
			sprintf(filename,"%s/ctf_radial_avg_raw.mrc",out_dir);
			Mrc::write(p_rad_avg,M,k,1,filename);
		}

		if( verbose >= 2 ) {
			sprintf(filename,"%s/ctf_radial_env_raw.mrc",out_dir);
			Mrc::write(p_rad_wgt,M,k,1,filename);
		}

		for(int n=0;n<k;n++) {
			p_nrm = p_rad_nrm + n*((int)M);
			p_avg = p_rad_avg + n*((int)M);
			p_wgt = p_rad_wgt + n*((int)M);
			adjust_radial_avg(p_nrm,p_avg,p_wgt);
			float dz = get_def_shift(c_def_inf[n].w,p_nrm,ix2def*(c_def_inf[n].x+c_def_inf[n].y)/2);
			c_def_inf[n].x = ix2def*c_def_inf[n].x + dz;
			c_def_inf[n].y = ix2def*c_def_inf[n].y + dz;
		}
		
		if( verbose >= 1 ) {
			sprintf(filename,"%s/ctf_radial_avg.mrc",out_dir);
			Mrc::write(p_rad_avg,M,k,1,filename);
		}

		if( verbose >= 1 ) {
			sprintf(filename,"%s/ctf_radial_envelope.mrc",out_dir);
			Mrc::write(p_rad_wgt,M,k,1,filename);
		}
		
		if( verbose >= 2 ) {
			sprintf(filename,"%s/ctf_radial_norm.mrc",out_dir);
			Mrc::write(p_rad_nrm,M,k,1,filename);
		}
	}
	
	void adjust_radial_avg(float*p_nrm,float*p_avg,float*p_wgt) {
		float wgt_max = 0;
		int min_m = (int)ceil(fpix_range.x);
		int max_m = (int)ceil(fpix_range.y);
		for(int m=0;m<M;m++) {
			p_wgt[m] = p_wgt[m]/M;
			p_avg[m] += p_wgt[m];
			if( m > min_m && m < max_m )
				wgt_max = max(wgt_max,p_wgt[m]);
			float den = 2*p_wgt[m];
			if( abs(den) < SUSAN_FLOAT_TOL ) {
				if( den < 0 ) den = -1;
				else den = 1;
			}
			p_nrm[m] = p_avg[m]/den;
		}
		
		for(int m=1;m<(M-1);m++) {
			if( abs(p_nrm[m]) > 1 ) {
				p_nrm[m] = (p_nrm[m-1]+p_nrm[m+1])/2;
			}
		}
		
		for(int m=0;m<min_m;m++) {;
			p_avg[m] = p_avg[m]/(2*p_wgt[m]);
			p_wgt[m] = 1;
		}

		for(int m=min_m;m<M;m++) {
			p_wgt[m] = max(min(p_wgt[m]/wgt_max,1.0),0.0);
			p_avg[m] = max(min(p_avg[m]/(2*wgt_max),1.0),0.0);
		}
		
                for(int m=0;m<M;m++) {
			p_avg[m] = p_avg[m]*(sqrt(p_wgt[m]))/p_wgt[m];
			p_wgt[m] = (sqrt(p_wgt[m]));
                }
	}
	
	float get_def_shift(single&fit_coef,const single*p_in,const float dz_base) {
		int t_min = ceilf(fpix_range.x);
		int t_max = ceilf(fpix_range.y);
		
		float rslt = 0;
		float val = l1_def(p_in,dz_base,t_min,t_max);
		
		for(float dz=-ref_range;dz<=ref_range;dz+=ref_step) {
			float cur_val = l1_def(p_in,dz_base+dz,t_min,t_max);
			if(cur_val<val) {
				val = cur_val;
				rslt = dz;
			}
		}
		
		fit_coef = val;
		
		return rslt;
	}
	
	float l1_def(const single*p_in,const float def,const int t_min,const int t_max) {
		float v = 0;
		
		for(int t=t_min;t<t_max;t++) {
			float s2 = (float)t;
			s2 = s2/(N*apix);
			s2 = s2*s2;
			float s4 = s2*s2;
			float gamma = lambda_pi*def*s2 - lambda3_Cs_pi_2*s4;
			float ctf = CA*sin(gamma) + AC*cos(gamma);
			v += abs( p_in[t] - ctf*ctf );
		}
		
		return (v/(t_max-t_min));
	}
	
	void generate_ctf(const int k) {
		for(int i=0;i<k;i++) {
			float def = (c_def_inf[i].x+c_def_inf[i].y)/2;
			float*Pctf = p_ctf_fit + i*((int)M);
			for(int j=0;j<M;j++) {
				float s2 = (float)j;
				s2 = s2/(N*apix);
				s2 = s2*s2;
				float s4 = s2*s2;
				float gamma = lambda_pi*def*s2 - lambda3_Cs_pi_2*s4;
				float ctf = CA*sin(gamma) + AC*cos(gamma);
				Pctf[j] = ctf*ctf;
			}
		}
	}
	
	void generate_phase_dif(const int k) {
		for(int i=0;i<k;i++) {
			int off = i*((int)M);
			calc_phase_dif(p_ctf_res+off,p_ctf_fit+off,p_rad_nrm+off);
		}
	}
	
	void calc_phase_dif(float*p_out,const float*p_ctf,const float*p_ref) {
		int min_j = (int)ceil(fpix_range.x);
		for(int j=0;j<M;j++) {
			p_out[j] = -1;
			if( j > min_j && j < (M-4) ) {
				if( p_ctf[j] > 0.8 ) {
					float cur_max = p_ctf[j];
					for(int k=-2;k<3;k++) {
						cur_max = max(cur_max,p_ctf[j+k]);
					}
					if( p_ctf[j] == cur_max ) {
						p_out[j] = 1-abs( p_ctf[j] - p_ref[j] );
					}
				}
			}
		}
	}
	
	void save_defocus(const int k,const char*out_dir) {
		sprintf(filename,"%s/defocus.txt",out_dir);
		FILE*fp = fopen(filename,"w");
		Defocus def;
		def.ph_shft = 0;
		def.ExpFilt = 0;

                for(int i=0;i<k;i++) {
                    int idx = i*(int)M;
                    estimate_params(def.Bfactor,def.max_res,p_rad_avg+idx,p_ctf_res+idx);
                    def.U     = c_def_inf[i].x;
                    def.V     = c_def_inf[i].y;
                    def.angle = c_def_inf[i].z*180.0/M_PI;
                    def.score = c_def_inf[i].w;
                    IO::DefocusIO::write(fp,def);
                }

                fclose(fp);

	}

	void save_svg_report(const int k,const char*out_dir,const int req_verb) {
		
		if( verbose >= req_verb ) {
			sprintf(filename,"%s/ctf_fit",out_dir);
			IO::create_dir(filename);
			for(int i=0;i<k;i++) {
				float def = (c_def_inf[i].x+c_def_inf[i].y)/2;
				int off = i*(int(M));
				sprintf(filename,"%s/ctf_fit/projection_%03d.svg",out_dir,i+1);
				SvgCtf report(filename,apix);
				report.create_grid(fpix_range.x,fpix_range.y,N);
				report.create_title(i+1,def);
				report.add_est(p_ctf_res+off,M);
				report.add_avg(p_rad_avg+off,M);
				report.add_fit(p_ctf_fit+off,M);
				report.create_legend();
			}
		}
	}

	void save_fitted_defocus(const int k,const char*out_dir) {
		
		/*float wgt_max = 0;
		int min_m = (int)ceil(fpix_range.x);
		int max_m = (int)ceil(fpix_range.y);
		for(int m=0;m<M;m++) {
			p_wgt[m] = p_wgt[m]/M;
			p_avg[m] += p_wgt[m];
			if( m > min_m && m < max_m )
				wgt_max = max(wgt_max,p_wgt[m]);
			float den = 2*p_wgt[m];
			if( abs(den) < SUSAN_FLOAT_TOL ) {
				if( den < 0 ) den = -1;
				else den = 1;
			}
			p_nrm[m] = p_avg[m]/den;
		}
		
		for(int m=1;m<(M-1);m++) {
			if( abs(p_nrm[m]) > 1 ) {
				p_nrm[m] = (p_nrm[m-1]+p_nrm[m+1])/2;
			}
		}
		
		for(int m=0;m<min_m;m++) {;
			p_avg[m] = p_avg[m]/(2*p_wgt[m]);
			p_wgt[m] = 1;
		}

		for(int m=min_m;m<M;m++) {
			p_wgt[m] = max(min(p_wgt[m]/wgt_max,1.0),0.0);
			p_avg[m] = max(min(p_avg[m]/(2*wgt_max),1.0),0.0);
                //}
		
		int min_m = (int)ceil(fpix_range.x);
		int max_m = (int)ceil(fpix_range.y);
		
		for(int i=0;i<k;i++) {
			int off = i*(int)(M);
			float*p_work = p_rad_wgt + off;
			float wgt_max = 0;
			for(int t=0;t<M;t++) {
				p_work[t] = p_work[t]/M;
				if( t > min_m && t < max_m )
					wgt_max = max(wgt_max,p_work[t]);
			}
			for(int t=0;t<min_m;t++) {
				p_work[t] = p_work[t];
			}
			for(int t=min_m;t<M;t++) {
				p_work[t] = wgt_max;
			}
		}
		
		cudaMemcpy( (void*)rad_wgt.ptr, (const void*)(p_rad_wgt), sizeof(float)*ss_2.x*ss_2.y, cudaMemcpyHostToDevice);
		cudaMemcpy( (void*)g_def_inf.ptr, (const void*)(c_def_inf), sizeof(float4)*k, cudaMemcpyHostToDevice);
		
		float*output = new float[(int)round(N*N*K)];
		
                GpuKernelsCtf::normalize_amplitude<<<grd_c,blk>>>(ss_data_c.ptr,rad_wgt.ptr,ss_c);
		GpuKernelsCtf::vis_copy_data<<<grd_r,blk>>>(ss_vis.ptr,ss_data_c.ptr,ss_r);
		GpuKernelsCtf::vis_add_ctf<<<grd_r,blk>>>(ss_vis.ptr,g_def_inf.ptr,apix,lambda_pi,lambda3_Cs_pi_2,AC,ss_r);
		save_gpu_mrc(output,ss_vis.ptr,ss_r.x,ss_r.y,ss_r.z,out_dir,"ctf_fitting_result.mrc",0);
		
		delete [] output;
		
	}

	void estimate_params(float&Bfactor,float&max_res,const float*data,const float*peaks) {
		Bfactor = 0;
		int min_j = (int)ceil(fpix_range.x);
		int count=0;
                int max_j=0;
		float s2;
		for(int j=min_j;j<M;j++) {
			if( peaks[j] > 0 ) {
				count++;
				if(peaks[j]<res_thres) {
					max_j = j;
					break;
				}
			}
                }

                if(max_j>0)
                    max_res = N*apix/((float)max_j);
                else
                    max_res = 0;
		
		float2 *extracted_data = new float2[count];
		
                if( count > 2 && max_j > 0 ) {
			count=0;
			for(int j=min_j;j<max_j;j++) {
				if( peaks[j] > 0 ) {
					s2 = (float)j;
					s2 = s2/(N*apix);
					s2 = s2*s2;
					extracted_data[count].x = s2;
					extracted_data[count].y = data[j];
					count++;
				}
			}
			
			float2 bfac_val;
			bfac_val.x = 0;
			bfac_val.y = 9999999.9;
			for(float bfac=0;bfac<=bfac_max;bfac+=50) {
				float avg=0,std=0,val;
				for(int i=0;i<count;i++) {
					val = extracted_data[i].y/exp(-bfac*extracted_data[i].x/4);
					avg += val;
				}
				avg = avg/count;
				for(int i=0;i<count;i++) {
					val = extracted_data[i].y/exp(-bfac*extracted_data[i].x/4) - avg;
					std += val*val;
				}
				std = sqrt(std/count);
				
				if( std < bfac_val.y ) {
					bfac_val.y = std;
					bfac_val.x = bfac;
				}
			}
			
                        Bfactor = bfac_val.x;
		}

                delete [] extracted_data;
		
        }*/

};

#endif /// CTF_LINEARIZER_H


