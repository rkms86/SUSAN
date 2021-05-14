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
public:
	int3  ss_c;
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
	GPU::GArrSingle   ss_vis;
	GPU::GArrSingle2  ss_data_R;
	GPU::GArrSingle3  ss_filter;
	GPU::GArrSingle4  g_def_inf;
	GPU::GArrSingle   rad_avg;
	GPU::GArrSingle   rad_wgt;
	GPU::GArrSingle2  rad_fourier;
	GPU::GArrSingle2  rad_hilbert;
	GpuFFT::FFT2D     fft2;
	GPU::GTex2DSingle ss_lin;
	
	float  *p_ps_lin_avg;
	float  *p_rad_avg;
	float  *p_rad_wgt;
	float  *p_rad_nrm;
	float  *p_ctf_fit;
	float  *p_ctf_res;
	float4 *c_def_inf;
		
	
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
		
		ss_vis.alloc(N*N*K);
		ss_input.alloc(M*N*K);
		ss_sum_z.alloc(M*N);
		ss_data_c.alloc(M*N*K);
		ss_data_r.alloc(N*N*K);
		ss_data_R.alloc(M*N*K);
		
		g_def_inf.alloc(K);
		
		rad_avg.alloc(M*K);
		rad_wgt.alloc(M*K);
		rad_fourier.alloc(M*K);
		rad_hilbert.alloc(M*K);
		
		ss_lin.alloc(M,N,K);
		fft2.alloc(M,N,K);
		
		create_filter();
		
		p_ps_lin_avg = new float[int(M*N)];
		c_def_inf = new float4[int(K)];
		p_rad_avg = new float[int(M*K)];
		p_rad_wgt = new float[int(M*K)];
		p_rad_nrm = new float[int(M*K)];
		p_ctf_fit = new float[int(M*K)];
		p_ctf_res = new float[int(M*K)];
	}
	
	~CtfLinearizer() {
		delete [] p_ps_lin_avg;
		delete [] c_def_inf;
		delete [] p_rad_avg;
		delete [] p_rad_wgt;
		delete [] p_rad_nrm;
		delete [] p_ctf_fit;
		delete [] p_ctf_res;
	}
	
	void load_info(ArgsCTF::Info*info,Tomogram*p_tomo) {
		
		ss_c  = make_int3(M,N,p_tomo->num_proj);
		ss_r  = make_int3(N,N,p_tomo->num_proj);
		ss_2  = make_int3(M,p_tomo->num_proj,1);
		
		blk   = GPU::get_block_size_2D();
		grd_c = GPU::calc_grid_size(blk,ss_c.x,ss_c.y,ss_c.z);
		grd_r = GPU::calc_grid_size(blk,ss_r.x,ss_r.y,ss_r.z);
		grd_2 = GPU::calc_grid_size(blk,ss_2.x,ss_2.y,ss_2.z);
		
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
		
		tlt_fpix = ceilf(info->tlt_range/ix2def);
	}
	
	void process(const char*out_dir,float*input,Tomogram*p_tomo) {
		
		rad_avg.clear();
		rad_wgt.clear();
		
		cudaMemcpy( (void*)ss_input.ptr, (const void*)(input), sizeof(float)*ss_c.x*ss_c.y*ss_c.z, cudaMemcpyHostToDevice);
		GpuKernelsCtf::rmv_bg<<<grd_c,blk>>>(ss_data_c.ptr,ss_input.ptr,ss_filter.ptr,n_filter,ss_c);
		GpuKernelsCtf::keep_fpix_range<<<grd_c,blk>>>(ss_data_c.ptr,fpix_range,ss_c);

		save_gpu_mrc(input,ss_data_c.ptr,ss_c.x,ss_c.y,ss_c.z,out_dir,"ctf_normalized.mrc",1);
		
		GpuKernels::load_surf<<<grd_c,blk>>>(ss_lin.surface,ss_data_c.ptr,ss_c);
		GpuKernelsCtf::ctf_linearize<<<grd_c,blk>>>(ss_data_c.ptr,ss_lin.texture,linearization_scale,ss_c);
		GpuKernels::load_surf<<<grd_c,blk>>>(ss_lin.surface,ss_data_c.ptr,ss_c);
		GpuKernelsCtf::tangential_blur<<<grd_c,blk>>>(ss_data_c.ptr,ss_lin.texture,ss_c);
		
		save_gpu_mrc(input,ss_data_c.ptr,ss_c.x,ss_c.y,ss_c.z,out_dir,"ctf_linearized.mrc",2);
		
		GpuKernels::expand_ps_hermitian<<<grd_r,blk>>>(ss_data_r.ptr,ss_data_c.ptr,ss_r);
		GpuKernels::fftshift2D<<<grd_r,blk>>>(ss_data_r.ptr,ss_r);
		fft2.exec(ss_data_R.ptr,ss_data_r.ptr);
		GpuKernels::fftshift2D<<<grd_c,blk>>>(ss_data_R.ptr,ss_c);
		GpuKernels::load_surf_abs<<<grd_c,blk>>>(ss_lin.surface,ss_data_R.ptr,ss_c);
		GpuKernelsCtf::tangential_blur<<<grd_c,blk>>>(ss_data_c.ptr,ss_lin.texture,ss_c);
		GpuKernels::load_surf<<<grd_c,blk>>>(ss_lin.surface,ss_data_c.ptr,ss_c);
		GpuKernelsCtf::radial_highpass<<<grd_c,blk>>>(ss_data_c.ptr,ss_lin.texture,ss_c);
		GpuKernelsCtf::keep_fpix_range<<<grd_c,blk>>>(ss_data_c.ptr,def_lin_range,ss_c);
		
		save_gpu_mrc(input,ss_data_c.ptr,ss_c.x,ss_c.y,ss_c.z,out_dir,"ctf_ps_lin_raw.mrc",1);
		
		GpuKernelsCtf::sum_along_z<<<grd_c,blk>>>(ss_sum_z.ptr,ss_data_c.ptr,ss_c);
		
		float3 avg_def;
		save_gpu_mrc(p_ps_lin_avg,ss_sum_z.ptr,ss_c.x,ss_c.y,1,out_dir,"ctf_ps_lin_avg.mrc",1);
		get_avg_defocus(avg_def,p_ps_lin_avg);
		set_def_info(avg_def,p_tomo);
		printf("          - Average defocus (Angstroms): U=%.2f, V=%.2f, angle=%.1f (Linear fourier pixel size: %.1f)\n",avg_def.x*ix2def,avg_def.y*ix2def,avg_def.z*180.0/M_PI,ix2def);
		
		GpuKernelsCtf::mask_ellipsoid<<<grd_c,blk>>>(ss_data_c.ptr,g_def_inf.ptr,ss_c);
		save_gpu_mrc(input,ss_data_c.ptr,ss_c.x,ss_c.y,ss_c.z,out_dir,"ctf_ps_lin.mrc",2);
		
		int n_avg_f = 8;
		float2 tmp_range;
		tmp_range.x = n_avg_f;
		tmp_range.y = M-n_avg_f-1;
		float lambda_def = ix2def*lambda*(avg_def.x+avg_def.y)/2;
		get_initial_defocus(avg_def,input);
		GpuKernelsCtf::rmv_bg<<<grd_c,blk>>>(ss_data_c.ptr,ss_input.ptr,ss_filter.ptr,n_filter,ss_c);
		GpuKernelsCtf::keep_fpix_range<<<grd_c,blk>>>(ss_data_c.ptr,tmp_range,ss_c);
		GpuKernels::load_surf<<<grd_c,blk>>>(ss_lin.surface,ss_data_c.ptr,ss_c);
		GpuKernelsCtf::ctf_radial_normalize<<<grd_c,blk>>>(ss_data_c.ptr,ss_lin.texture,g_def_inf.ptr,M_PI*lambda,apix,ss_c);
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
		GpuKernelsCtf::rmv_bg<<<grd_c,blk>>>(ss_data_c.ptr,ss_input.ptr,ss_filter.ptr,n_filter,ss_c);
		GpuKernelsCtf::keep_fpix_range<<<grd_c,blk>>>(ss_data_c.ptr,tmp_range,ss_c);
		save_fitted_defocus(ss_2.y,out_dir);
	}
	
protected:
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
	
	void save_gpu_mrc(single*p_cpu,const single*p_gpu,const int m,const int n,const int k,const char*out_dir,const char*name,const int req_verb) {
		cudaMemcpy((void*)p_cpu, (const void*)p_gpu, sizeof(float)*m*n*k, cudaMemcpyDeviceToHost);
		if( verbose >= req_verb ) {
			sprintf(filename,"%s/%s",out_dir,name);
			Mrc::write(p_cpu,m,n,k,filename);
		}
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
	
	void set_def_info(const float3&defocus,Tomogram*p_tomo) {
		float L = (defocus.x+defocus.y)/2;
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
		for(int m=0;m<M;m++) {
			p_wgt[m] = p_wgt[m]/M;
			p_avg[m] += p_wgt[m];
			wgt_max = max(wgt_max,p_wgt[m]);
			float den = 2*p_wgt[m];
			if( abs(den) < SUSAN_FLOAT_TOL ) {
				if( den < 0 ) den = -1;
				else den = 1;
			}
			p_nrm[m] = p_avg[m]/den;
		}
		for(int m=0;m<M;m++) {
			p_wgt[m] = p_wgt[m]/wgt_max;
			p_avg[m] = p_avg[m]/(2*wgt_max);
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
		int t_min = ceilf(fpix_range.x);
		for(int i=0;i<k;i++) {
			int off = i*(int)(M);
			float*p_work = p_rad_wgt + off;
			for(int t=0;t<M;t++) {
				p_work[t] = p_work[t]/M;
			}
			float max_val = p_work[t_min];
			for(int t=t_min;t<M;t++) {
				max_val = max(p_work[t],max_val);
			}
			for(int t=t_min;t<M;t++) {
				p_work[t]=max_val;
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
		
	}

};

#endif /// CTF_LINEARIZER_H


