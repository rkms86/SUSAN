#ifndef REF_ALI_H
#define REF_ALI_H

#include "datatypes.h"
#include "particles.h"
#include "tomogram.h"
#include "stack_reader.h"
#include "gpu.h"
#include "gpu_fft.h"
#include "gpu_rand.h"
#include "gpu_kernel.h"
#include "gpu_kernel_ctf.h"
#include "gpu_kernel_vol.h"
#include "substack_crop.h"
#include "mrc.h"
#include "io.h"
#include "points_provider.h"
#include "angles_symmetry.h"
#include "aligner_args.h"
#include <iostream>

class AliSubstack {
public:
	int N;
	int M;
	int P;
	int maxK;
	int NP;
	int MP;

	GPU::GArrSingle  ss_padded;
	GPU::GArrSingle3 ss_vec_r;
	GPU::GArrSingle2 ss_fourier;
	GPU::GArrSingle  rad_avg;
	GPU::GArrSingle  rad_wgt;

	GpuFFT::FFT2D          fft2;
	GpuRand::BoxMullerRand rand;
	
	AliSubstack(int m,int n,int k,int p,GPU::Stream&stream) : rand(n+p,n+p,k) {
		N  = n;
		M  = m;
		P  = p;
		NP = n+p;
		MP = (NP/2)+1;
		maxK = k;
		
		ss_padded.alloc(NP*NP*maxK);
		ss_fourier.alloc(MP*NP*maxK);

		rad_avg.alloc(MP*maxK);
		rad_wgt.alloc(MP*maxK);

		ss_vec_r.alloc(MP*NP);
		
		int3 ss_siz = make_int3(MP,NP,1);
		dim3 blk2D = GPU::get_block_size_2D();
		dim3 grd2D = GPU::calc_grid_size(blk2D,ss_siz.x,ss_siz.y,ss_siz.z);
		GpuKernelsCtf::create_vec_r<<<grd2D,blk2D,0,stream.strm>>>(ss_vec_r.ptr,ss_siz);
		
		fft2.alloc(MP,NP,maxK);
		fft2.set_stream(stream.strm);
	}
	
	void pad_zero(GPU::Stream&stream) {
		ss_padded.clear(stream.strm);
	}
	
	void pad_normal(GPU::GArrSingle2&g_avg_std,int k,GPU::Stream&stream) {
		int3 ss_pad = make_int3(NP,NP,k);
		rand.gen_normal(ss_padded.ptr,g_avg_std.ptr,ss_pad,stream.strm);
	}
	
	void add_data(GPU::GArrSingle&g_data,GPU::GArrProj2D&g_ali,int k,GPU::Stream&stream) {
		int3 pad = make_int3(P/2,P/2,0);
		int3 ss_raw = make_int3(N,N,k);
		int3 ss_pad = make_int3(NP,NP,k);
		int3 ss_fou = make_int3(MP,NP,k);
		dim3 blk = GPU::get_block_size_2D();
		dim3 grd_raw = GPU::calc_grid_size(blk,N,N,k);
		dim3 grd_pad = GPU::calc_grid_size(blk,NP,NP,k);
		dim3 grd_fou = GPU::calc_grid_size(blk,MP,NP,k);
		GpuKernels::load_pad<<<grd_raw,blk,0,stream.strm>>>(ss_padded.ptr,g_data.ptr,pad,ss_raw,ss_pad);
		GpuKernels::fftshift2D<<<grd_pad,blk,0,stream.strm>>>(ss_padded.ptr,ss_pad);
                fft2.exec(ss_fourier.ptr,ss_padded.ptr);
                GpuKernels::fftshift2D<<<grd_fou,blk,0,stream.strm>>>(ss_fourier.ptr,ss_fou);
		GpuKernels::subpixel_shift<<<grd_fou,blk,0,stream.strm>>>(ss_fourier.ptr,g_ali.ptr,ss_fou);
	}
	
	void correct_wiener(const CtfConst ctf_const,GPU::GArrSingle&ctf_wgt,GPU::GArrDefocus&g_def,float3 bandpass,int k,GPU::Stream&stream) {
		int3 ss = make_int3(MP,NP,k);
		dim3 blk = GPU::get_block_size_2D();
		dim3 grd = GPU::calc_grid_size(blk,MP,NP,k);
                GpuKernelsCtf::correct_stk_wiener<<<grd,blk,0,stream.strm>>>(ss_fourier.ptr,ctf_wgt.ptr,g_def.ptr,bandpass,ctf_const,ss);
	}

	void correct_wiener_ssnr(const CtfConst ctf_const,GPU::GArrSingle&ctf_wgt,GPU::GArrDefocus&g_def,float3 bandpass,float2 ssnr,int k,GPU::Stream&stream) {
		// float2  ssnr; /// x=F; y=S;
		single ssnr_f = -100*ssnr.x/(NP*ctf_const.apix);
		single ssnr_s = pow(10,3*ssnr.y);
		int3 ss = make_int3(MP,NP,k);
		dim3 blk = GPU::get_block_size_2D();
		dim3 grd = GPU::calc_grid_size(blk,MP,NP,k);
		GpuKernelsCtf::correct_stk_wiener_ssnr<<<grd,blk,0,stream.strm>>>(ss_fourier.ptr,ctf_wgt.ptr,g_def.ptr,ssnr_f,ssnr_s,bandpass,ctf_const,ss);
        }

        void whitening_filter(int k,GPU::Stream&stream) {
                whitening_filter(ss_fourier,k,stream);
        }

        void whitening_filter(GPU::GArrSingle2&ss_data,int k,GPU::Stream&stream) {
                rad_avg.clear(stream.strm);
                rad_wgt.clear(stream.strm);
                int3 ss = make_int3(MP,NP,k);
                dim3 blk = GPU::get_block_size_2D();
                dim3 grd = GPU::calc_grid_size(blk,MP,NP,k);
                GpuKernels::radial_ps_avg<<<grd,blk,0,stream.strm>>>(rad_avg.ptr,rad_wgt.ptr,ss_data.ptr,ss);
                GpuKernels::radial_ps_norm<<<grd,blk,0,stream.strm>>>(ss_data.ptr,rad_avg.ptr,rad_wgt.ptr,ss);
        }

        void norm_complex(GPU::GArrSingle2&ss_data,int k,GPU::Stream&stream) {
                int3 ss = make_int3(MP,NP,k);
                dim3 blk = GPU::get_block_size_2D();
                dim3 grd = GPU::calc_grid_size(blk,MP,NP,k);
                GpuKernels::norm_complex<<<grd,blk,0,stream.strm>>>(ss_data.ptr,ss);
                GpuKernels::apply_radial_wgt<<<grd,blk,0,stream.strm>>>(ss_data.ptr,MP,ss);
        }

	void apply_radial_wgt(float w_total,int k,GPU::Stream&stream) {
		int3 ss = make_int3(MP,NP,k);
		dim3 blk = GPU::get_block_size_2D();
		dim3 grd = GPU::calc_grid_size(blk,MP,NP,k);
                GpuKernels::apply_radial_wgt<<<grd,blk,0,stream.strm>>>(ss_fourier.ptr,w_total,ss);
	}

};

class AliData {
public:
    Vec3           *c_pts;
    GPU::GArrProj2D g_ali;
    GPU::GArrVec3   g_pts;
    GPU::GArrSingle g_cc;
    float          *c_cc;
    uint32 M;
    uint32 N;
    uint32 max_K;
    uint32 n_pts;

    GPU::GArrSingle2 prj_c;
    GPU::GArrSingle  prj_r;

    GpuFFT::IFFT2D ifft2;

    GPU::GTex2DSingle prj_tex;

    AliData(uint32 m, uint32 n, uint32 n_K,const float4&off_params,int off_type,GPU::Stream&stream) {
        M = m;
        N = n;
        max_K = n_K;
        g_ali.alloc(max_K);
        prj_c.alloc(M*N*max_K);
        prj_r.alloc(N*N*max_K);

        prj_tex.alloc(n,n,n_K);
        
        ifft2.alloc(M,N,max_K);
        ifft2.set_stream(stream.strm);

        if( off_type == ArgsAli::OffsetType_t::ELLIPSOID )
            c_pts = PointsProvider::ellipsoid(n_pts,off_params.x,off_params.y,off_params.z,off_params.w);
        if( off_type == ArgsAli::OffsetType_t::CYLINDER )
            c_pts = PointsProvider::cylinder(n_pts,off_params.x,off_params.y,off_params.z,off_params.w);
        if( off_type == ArgsAli::OffsetType_t::CIRCLE )
            c_pts = PointsProvider::circle(n_pts,off_params.x,off_params.y);

        if( off_type == ArgsAli::OffsetType_t::CIRCLE ) {
            c_cc = new float[n_pts*max_K];
            g_cc.alloc(n_pts*max_K);
        }
        else {
            c_cc = new float[n_pts];
            g_cc.alloc(n_pts);
        }
        g_pts.alloc(n_pts);
        cudaMemcpy((void*)g_pts.ptr,(const void*)c_pts,sizeof(Vec3)*n_pts,cudaMemcpyHostToDevice);
    }

    ~AliData() {
        delete [] c_cc;
        delete [] c_pts;
    }

    void rotate_post(Rot33&R,GPU::GArrProj2D&ali_in,int k,GPU::Stream&stream) {
        dim3 blk;
        dim3 grd;
        blk.x = 1024;
        blk.y = 1;
        blk.z = 1;
        grd.x = GPU::div_round_up(9*k,1024);
        grd.y = 1;
        grd.z = 1;

        GpuKernels::rotate_post<<<grd,blk,0,stream.strm>>>(g_ali.ptr,R,ali_in.ptr,k);
    }

    void rotate_pre(Rot33&R,GPU::GArrProj2D&ali_in,int k,GPU::Stream&stream) {
        dim3 blk;
        dim3 grd;
        blk.x = 1024;
        blk.y = 1;
        blk.z = 1;
        grd.x = GPU::div_round_up(9*k,1024);
        grd.y = 1;
        grd.z = 1;

        GpuKernels::rotate_pre<<<grd,blk,0,stream.strm>>>(g_ali.ptr,R,ali_in.ptr,k);
    }

    void project(GPU::GTex3DSingle2&ref,float3 bandpass,int k,GPU::Stream&stream) {
        int3 ss_fou = make_int3(M,N,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,M,N,k);
        GpuKernelsVol::extract_stk<<<grd,blk,0,stream.strm>>>(prj_c.ptr,ref.surface,g_ali.ptr,bandpass,M,N,k);
        GpuKernels::divide<<<grd,blk,0,stream.strm>>>(prj_c.ptr,N*N*N,ss_fou);
        GpuKernels::divide<<<grd,blk,0,stream.strm>>>(prj_c.ptr,N*N,ss_fou);
    }

    void invert_fourier(int k,GPU::Stream&stream) {
        int3 ss_fou = make_int3(M,N,k);
        int3 ss_pad = make_int3(N,N,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd_f = GPU::calc_grid_size(blk,M,N,k);
        dim3 grd_r = GPU::calc_grid_size(blk,N,N,k);
        GpuKernels::fftshift2D<<<grd_f,blk,0,stream.strm>>>(prj_c.ptr,ss_fou);
        ifft2.exec(prj_r.ptr,prj_c.ptr);
        GpuKernels::fftshift2D<<<grd_r,blk,0,stream.strm>>>(prj_r.ptr,ss_pad);
    }

    void multiply(GPU::GArrSingle&p_wgt,int k,GPU::Stream&stream) {
        int3 ss = make_int3(M,N,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,M,N,k);
        GpuKernels::multiply<<<grd,blk,0,stream.strm>>>(prj_c.ptr,p_wgt.ptr,ss);
    }

    void multiply(GPU::GArrSingle2&p_data,int k,GPU::Stream&stream) {
        int3 ss = make_int3(M,N,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,M,N,k);
        GpuKernels::multiply<<<grd,blk,0,stream.strm>>>(prj_c.ptr,p_data.ptr,ss);
    }

    void sparse_reconstruct(GPU::GArrProj2D&ali,int k,GPU::Stream&stream) {
        int3 ss = make_int3(N,N,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,N,N,k);
        GpuKernels::load_surf<<<grd,blk,0,stream.strm>>>(prj_tex.surface,prj_r.ptr,ss);

        blk.x = 1024;
        blk.y = 1;
        blk.z = 1;
        grd.x = GPU::div_round_up(n_pts,1024);
        grd.y = 1;
        grd.z = 1;
        GpuKernelsVol::reconstruct_pts<<<grd,blk,0,stream.strm>>>(g_cc.ptr,ali.ptr,prj_tex.texture,g_pts.ptr,n_pts,N,k);
        GPU::download_async(c_cc,g_cc.ptr,n_pts,stream.strm);
    }

    void extract_cc(float*p_cc,int*p_ix,int k,GPU::Stream&stream) {
        dim3 blk;
        dim3 grd;
        blk.x = 1024;
        blk.y = 1;
        blk.z = 1;
        grd.x = GPU::div_round_up(n_pts,1024);
        grd.y = 1;
        grd.z = k;
        GpuKernelsVol::extract_pts<<<grd,blk,0,stream.strm>>>(g_cc.ptr,prj_r.ptr,g_pts.ptr,n_pts,N,k);
        GPU::download_async(c_cc,g_cc.ptr,n_pts*k,stream.strm);
        stream.sync();
        for(int i=0;i<k;i++) {
            get_max_cc(p_cc[i],p_ix[i],c_cc+i*n_pts);
        }
    }

    void get_max_cc(float&max_cc,int&max_idx,const float*p_data) {
        max_cc = 0;
        max_idx = 0;
        for(int i=0;i<n_pts;i++) {
            if( max_cc < p_data[i] ) {
                max_cc = p_data[i];
                max_idx = i;
            }
        }
    }
};

class AliRef {
public:
    GPU::GTex3DSingle2 ref;
    int N;
    int M;

    void allocate(GPU::GArrSingle2&g_fou,int m,int n) {
        M = m;
        N = n;
        ref.alloc(M,N,N);

        bool should_conjugate = true;

        int3 siz = make_int3(M,N,N);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,M,N,N);
        GpuKernels::load_surf_3<<<grd,blk>>>(ref.surface,g_fou.ptr,siz,should_conjugate);
    }

};

/*
class RecAcc {
	
public:
	int maxK;
	int NP;
	int MP;
	
	dim3 blk;
	dim3 grd;
	
	GPU::GArrDouble2 vol_acc;
	GPU::GArrDouble  vol_wgt;
	
	void alloc(const int x,const int y,const int z) {
		MP = x;
		NP = y;
		maxK = z;
		
		vol_acc.alloc(MP*NP*NP);
		vol_wgt.alloc(MP*NP*NP);
		
		vol_acc.clear();
		vol_wgt.clear();
		
		blk = GPU::get_block_size_2D();
		grd = GPU::calc_grid_size(blk,MP,NP,NP);
	}
	
	void clear() {
		vol_acc.clear();
		vol_wgt.clear();
	}
	
	void insert(GPU::GTex2DSingle2&ss_stk,GPU::GTex2DSingle&ss_wgt,GPU::GArrProj2D&g_ali,float3 bandpass,int k,GPU::Stream&stream) {
		GpuKernelsVol::insert_stk<<<grd,blk,0,stream.strm>>>(vol_acc.ptr,vol_wgt.ptr,ss_stk.texture,ss_wgt.texture,g_ali.ptr,bandpass,MP,NP,k);
	}
};

class RecInvWgt {
public:
	int N;
	int M;
	
	int3 siz;
	dim3 blk;
	dim3 grd;
	
	int   inv_iter;
	float inv_gstd;
	
	int n_krnl;
	GPU::GArrDouble  g_wgt;
	GPU::GArrDouble  g_tmp;
	GPU::GArrDouble  g_conv;
	GPU::GArrSingle4 g_krnl;
	
	RecInvWgt(int n,int m,int in_iter,float in_gstd) {
		N = n;
		M = m;
		siz = make_int3(M,N,N);
		blk = GPU::get_block_size_2D();
		grd = GPU::calc_grid_size(blk,M,N,N);
		
		n_krnl   = 0;
		inv_iter = in_iter;
		inv_gstd = in_gstd;
		
		create_kernel();
		alloc();
	}
	
	void invert(GPU::GArrDouble&vol_wgt) {
		if( inv_iter > 0 ) {
			/// gWgt = gVolWgt;
			/// gVolWgt = sphere(N/2,N);
			/// for ...
			///    pTemp   = gVolWgt.*pWgt;
			///    pConv   = convn(pTemp,pKernel);
			///    gVolWgt = gVolWgt./max( abs(pConv), 0.0000001 );
			cudaMemcpy( (void*)g_wgt.ptr, (const void*)(vol_wgt.ptr), sizeof(double)*N*N*M, cudaMemcpyDeviceToDevice);
			GpuKernelsVol::inv_wgt_ite_sphere<<<grd,blk>>>(vol_wgt.ptr,siz);
			for(int i=0;i<inv_iter;i++) {
				GpuKernelsVol::inv_wgt_ite_multiply<<<grd,blk>>>(g_tmp.ptr  , vol_wgt.ptr, g_wgt.ptr , siz);
				GpuKernelsVol::inv_wgt_ite_convolve<<<grd,blk>>>(g_conv.ptr , g_tmp.ptr  , g_krnl.ptr, n_krnl, siz);
				GpuKernelsVol::inv_wgt_ite_divide  <<<grd,blk>>>(vol_wgt.ptr, g_conv.ptr , siz);
			}
		}
		else
			GpuKernelsVol::invert_wgt<<<grd,blk>>>(vol_wgt.ptr,siz);
	}
	
protected:
	void create_kernel() {
		if( inv_iter > 0 ) {
			float work_std = 2*inv_gstd*inv_gstd;
			float acc = 0;
			uint32 tmp;
			Vec3*c_filt = PointsProvider::circle(tmp,2,2);
			n_krnl = tmp;
			float4 *c_krnl = new float4[n_krnl];
			for(int i=0;i<n_krnl;i++) {
				c_krnl[i].x = c_filt[i].x;
				c_krnl[i].y = c_filt[i].y;
				c_krnl[i].z = c_filt[i].z;
				c_krnl[i].w = c_krnl[i].x*c_krnl[i].x + c_krnl[i].y*c_krnl[i].y + c_krnl[i].z*c_krnl[i].z;
				c_krnl[i].w = exp(-c_krnl[i].w/work_std);
				acc += c_krnl[i].w;
			}
			for(int i=0;i<n_krnl;i++) {
				c_krnl[i].w = c_krnl[i].w/acc;
			}
			g_krnl.alloc(n_krnl);
			cudaMemcpy( (void*)g_krnl.ptr, (const void*)(c_krnl), sizeof(float4)*n_krnl, cudaMemcpyHostToDevice);
			delete [] c_krnl;
			delete [] c_filt;
		}
	}

	void alloc() {
		if( inv_iter > 0 ) {
			g_wgt.alloc(N*N*M);
			g_tmp.alloc(N*N*M);
			g_conv.alloc(N*N*M);
		}
	}
};

class RecInvVol {
public:
	GPU::GArrSingle2 vol_fou;
	GPU::GArrSingle  vol_pad;
	GpuFFT::IFFT3D ifft3;
	
	int NP;
	int MP;
	int N;
	
	RecInvVol(int n,int p) {
		N  = n;
		NP = n+p;
		MP = (NP/2) + 1;
		vol_fou.alloc(NP*NP*MP);
		vol_pad.alloc(NP*NP*NP);
		ifft3.alloc(NP);
	}
	
	void apply_inv_wgt(GPU::GArrDouble2&vol_acc,GPU::GArrDouble&vol_wgt) {
		int3 siz = make_int3(MP,NP,NP);
		dim3 blk = GPU::get_block_size_2D();
		dim3 grd = GPU::calc_grid_size(blk,MP,NP,NP);
		GpuKernels::multiply<<<grd,blk>>>(vol_fou.ptr,vol_acc.ptr,vol_wgt.ptr,siz,1.0/(double)(NP*NP*NP));
	}
	
	void invert_and_extract(GPU::GArrSingle&vol_out) {
		dim3 blk = GPU::get_block_size_2D();
		dim3 grd = GPU::calc_grid_size(blk,MP,NP,NP);
		GpuKernels::fftshift3D<<<grd,blk>>>(vol_fou.ptr,MP,NP);
		ifft3.exec(vol_pad.ptr,vol_fou.ptr);
		GpuKernels::fftshift3D<<<grd,blk>>>(vol_pad.ptr,NP);
		
                //dim3 grdP = GPU::calc_grid_size(blk,NP,NP,NP);
                //GpuKernelsVol::grid_correct<<<grdP,blk>>>(vol_pad.ptr,NP);
		
		int3 siz_raw = make_int3(N,N,N);
		int3 siz_pad = make_int3(NP,NP,NP);
		dim3 grd3 = GPU::calc_grid_size(blk,N,N,N);
		GpuKernels::remove_pad_vol<<<grd3,blk>>>(vol_out.ptr,vol_pad.ptr,(NP-N)/2,siz_raw,siz_pad);
	}
	
	
	
};

class RecSym {
	
public:
	int N;
	int M;
	
	uint32 num_angs;
	M33f*p_angs;
	
	int3 siz;
	dim3 blk;
	dim3 grd;
	
	GPU::GArrDouble2 t_val;
	GPU::GArrDouble  t_wgt;
	
	RecSym(const int x,const int y,const char*symmetry) {
		M = x;
		N = y;
		
		p_angs = AnglesSymmetry::get_rotation_list(num_angs,symmetry);
		
		siz = make_int3(M,N,N);
		blk = GPU::get_block_size_2D();
		grd = GPU::calc_grid_size(blk,M,N,N);
		
		if( num_angs > 1 ) {
			t_val.alloc(M*N*N);
			t_wgt.alloc(M*N*N);
		}
	}
	
	~RecSym() {
		delete [] p_angs;
	}
	
	void apply_sym(GPU::GArrDouble2&vol_acc,GPU::GArrDouble&vol_wgt) {
		if( num_angs > 1 ) {
			cudaMemcpy(t_val.ptr,vol_acc.ptr,sizeof(double2)*M*N*N,cudaMemcpyDeviceToDevice);
			cudaMemcpy(t_wgt.ptr,vol_wgt.ptr,sizeof(double )*M*N*N,cudaMemcpyDeviceToDevice);
			
			Rot33 Rsym;
			for(uint32 i=1;i<num_angs;i++) {
				Math::set(Rsym,p_angs[i]);
				V3f tmp;
				Math::Rmat_eZYZ(tmp,p_angs[i]);
				GpuKernelsVol::add_symmetry<<<grd,blk>>>(vol_acc.ptr,vol_wgt.ptr,t_val.ptr,t_wgt.ptr,Rsym,M,N);
			}
		}
		
	}
};
*/

#endif /// REF_ALI_H


