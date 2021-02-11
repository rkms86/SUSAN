#ifndef GPU_KERNEL_VOL_H
#define GPU_KERNEL_VOL_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

#include "datatypes.h"

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cufft.h"

#include "gpu.h"
#include "gpu_kernel.h"

using namespace GpuKernels;

namespace GpuKernelsVol {

__device__ void  rot_inv_pt(float&x,float&y,float&z,const Rot33&R,const Vec3&in) {
    x = R.xx*in.x + R.yx*in.y + R.zx*in.z;
    y = R.xy*in.x + R.yy*in.y + R.zy*in.z;
    z = R.xz*in.x + R.yz*in.y + R.zz*in.z;
}

__device__ float rot_inv_pt_Z(const Rot33&R,const Vec3&in) {
    return R.xz*in.x + R.yz*in.y + R.zz*in.z;
}

__device__ void  rot_inv_pt_XY(float&x,float&y,const Rot33&R,const Vec3&in) {
    x = R.xx*in.x + R.yx*in.y + R.zx*in.z;
    y = R.xy*in.x + R.yy*in.y + R.zy*in.z;
}

__device__ bool  get_mirror_index(int&x,int&y,int&z,const int M,const int N) {
	if( x < 0 ) {
		x = -x; 
		y = N - y; if( y >= N ) y = 0;
		z = N - z; if( z >= N ) z = 0;
	}
	
	if( y>=0 && z >= 0 && x<M && y<N && z<N )
		return true;
	else
		return false;
}

__device__ bool  get_mirror_index(int&x,int&y,int&z,bool&was_inverted,const int M,const int N) {
	was_inverted = (x<0) ? true : false;
	return get_mirror_index(x,y,z,M,N);
}

__device__ int get_ix_3d(bool&should_conj,const int x,const int y,const int z,const int M,const int N) {
	int wx=x;
	int wy=y;
	int wz=z;
	should_conj = false;
	
	if(x<0) {
		wx = -wx;
		wy = -wy;
		wz = -wz;
		should_conj = true;
	}
	
	wy += N/2;
	wz += N/2;
	
	return wx + wy*M + wz*M*N;
}

////////////////////////////////////////////////////////////////////////

__global__ void insert_stk(double2*p_acc,double*p_wgt,
                           cudaTextureObject_t ss_stk, cudaTextureObject_t ss_wgt, const Proj2D*pTlt,
                           const float3 bandpass,const int M, const int N, const int K)
{
	int3 ss_idx = get_th_idx();
	
    if( ss_idx.x < M && ss_idx.y < N && ss_idx.z < N ) {

        Vec3 pt;
        pt.x = ss_idx.x;
        pt.y = ss_idx.y - N/2;
        pt.z = ss_idx.z - N/2;

        float R = sqrt( pt.x*pt.x + pt.y*pt.y + pt.z*pt.z );
        float bp = get_bp_wgt(bandpass.x,bandpass.y,bandpass.z,R);

        if( bp > 0.05 ) {

            bool should_add = false;
            double  wgt = 0;
            double2 val = {0,0};
            float x,y,z;

            for(int k=0; k<K; k++ ) {
                if( pTlt[k].w > 0 ){
                    z = rot_inv_pt_Z(pTlt[k].R,pt);
                    if( z >= 0 && z< 1 ) {
                        should_add = true;
                        bool should_conj = false;
                        rot_inv_pt_XY(x,y,pTlt[k].R,pt);
                        if(x<0){
                            x = -x;
                            y = -y;
                            should_conj = true;
                        }
                        float2 read_stk = tex2DLayered<float2>(ss_stk, x+0.5, y+N/2+0.5, k);
                        if(should_conj) {
                            read_stk.y = -read_stk.y;
                        }
                        val.x += (1-z)*read_stk.x;
                        val.y += (1-z)*read_stk.y;
                        float  read_wgt = tex2DLayered<float >(ss_wgt, x+0.5, y+N/2+0.5, k);
                        wgt   += (1-z)*read_wgt;

                    }
                }
            }

            if( should_add ) {
				long idx = ss_idx.x + ss_idx.y*M + ss_idx.z*M*N;
                double2 tmp = p_acc[ idx ];
                tmp.x += val.x;
                tmp.y += val.y;
                p_acc[ idx ]  = tmp;
                p_wgt[ idx ] += wgt;
            }

        }
    }

}

__global__ void invert_wgt(double*p_data,const int3 ss_siz) {
	
	int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {
		
		long idx = get_3d_idx(ss_idx,ss_siz);
		
		double data = p_data[idx];
		
		if( abs(data) < 0.0001 ) {
			if( data<0 )
				data = -1.0;
			else
				data =  1.0;
		}
		
		p_data[idx] = 1/data;
	}
}

__global__ void inv_wgt_ite_sphere(double*p_vol_wgt,const int3 ss_siz) {
    
    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

		int center = ss_siz.y/2;

		float R = l2_distance(ss_idx.x,ss_idx.y-center,ss_idx.z-center);

        double out = (R < center) ? 1.0 : 0.0;
        p_vol_wgt[ get_3d_idx(ss_idx,ss_siz) ] = out;
    }
}

__global__ void inv_wgt_ite_multiply(double*p_tmp,const double*p_vol_wgt,const double*p_wgt,const int3 ss_siz) {
    
    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {
		long idx = get_3d_idx(ss_idx,ss_siz);
        double out = p_vol_wgt[idx]*p_wgt[idx];
        p_tmp[ idx ] = out;
    }
}

__global__ void inv_wgt_ite_convolve(double*p_conv,const double*p_tmp,const float4*p_krnl,const int n_krnl,const int3 ss_siz) {
    
    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {
		
		double out = 0;
		
		for(int i=0; i<n_krnl; i++ ) {
            int x = ss_idx.x + (int)(p_krnl[i].x);
            int y = ss_idx.y + (int)(p_krnl[i].y);
            int z = ss_idx.z + (int)(p_krnl[i].z);
            double w = p_krnl[i].w;
            
			if( get_mirror_index(x,y,z,ss_siz.x,ss_siz.y) ) {
				out += w*p_tmp[ get_3d_idx(x,y,z,ss_siz) ];
			}
        }
		
        p_conv[ get_3d_idx(ss_idx,ss_siz) ] = out;
    }
}

__global__ void inv_wgt_ite_divide(double*p_vol_wgt, const double*p_conv,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {
		long idx = get_3d_idx(ss_idx,ss_siz);
        double den = p_conv[idx];
        den = fmax(den,1e-5);
        p_vol_wgt[ idx ] = fmin(p_vol_wgt[ idx ] / den, 1e10);
    }
}

__global__ void grid_correct(float*p_data,const int N) {

	int3 ss_idx = get_th_idx();

    if( ss_idx.x < N && ss_idx.y < N && ss_idx.z < N ) {

		long ix = ss_idx.x + ss_idx.y*N + ss_idx.z*N*N;

		int center = N/2;

		float R = l2_distance(ss_idx.x-center,ss_idx.y-center,ss_idx.z-center);
		
		float arg = fminf(R/N,0.5);
        arg = arg*M_PI;
        float sinc_coef = ( arg > 0.00001 ) ? sinf(arg)/arg : 1.0;
        sinc_coef *= sinc_coef;

        float val = p_data[ix];
        p_data[ix] = val/sinc_coef;
	}
}

__global__ void add_symmetry(double2*p_val,double*p_wgt,
                             const double2*t_val, const double*t_wgt,
                             Rot33 Rsym, const int M, const int N)
{
	int3 ss_idx = get_th_idx();
	
    if( ss_idx.x < M && ss_idx.y < N && ss_idx.z < N ) {

        Vec3 pt;
        pt.x = ss_idx.x;
        pt.y = ss_idx.y - N/2;
        pt.z = ss_idx.z - N/2;

        float R = sqrt( pt.x*pt.x + pt.y*pt.y + pt.z*pt.z );

        if( R < (N/2) ) {

			long idx = ss_idx.x + ss_idx.y*M + ss_idx.z*M*N;
            double2 val = p_val[idx];
            double  wgt = p_wgt[idx];
            float x,y,z;

			bool should_conj = false;
			
			rot_inv_pt(x,y,z,Rsym,pt);
			
			int p_x = floor(x);
			int p_y = floor(y);
			int p_z = floor(z);
			float w_x = x - floor(x);
			float w_y = y - floor(y);
			float w_z = z - floor(z);
			
			int     read_idx;
			double2 read_val;
			double  read_wgt;
			float   w;
			
			read_idx = get_ix_3d(should_conj,p_x  ,p_y  ,p_z  ,M,N);
			read_val = t_val[read_idx];
			read_wgt = t_wgt[read_idx];
			if(should_conj) read_val.y = -read_val.y;
			w = (1-w_x)*(1-w_y)*(1-w_z);
			val.x += w*read_val.x;
			val.y += w*read_val.y;
			wgt   += w*read_wgt;
			
			read_idx = get_ix_3d(should_conj,p_x+1,p_y  ,p_z  ,M,N);
			read_val = t_val[read_idx];
			read_wgt = t_wgt[read_idx];
			if(should_conj) read_val.y = -read_val.y;
			w = (  w_x)*(1-w_y)*(1-w_z);
			val.x += w*read_val.x;
			val.y += w*read_val.y;
			wgt   += w*read_wgt;
			
			read_idx = get_ix_3d(should_conj,p_x  ,p_y+1,p_z  ,M,N);
			read_val = t_val[read_idx];
			read_wgt = t_wgt[read_idx];
			if(should_conj) read_val.y = -read_val.y;
			w = (1-w_x)*(  w_y)*(1-w_z);
			val.x += w*read_val.x;
			val.y += w*read_val.y;
			wgt   += w*read_wgt;
			
			read_idx = get_ix_3d(should_conj,p_x+1,p_y+1,p_z  ,M,N);
			read_val = t_val[read_idx];
			read_wgt = t_wgt[read_idx];
			if(should_conj) read_val.y = -read_val.y;
			w = (  w_x)*(  w_y)*(1-w_z);
			val.x += w*read_val.x;
			val.y += w*read_val.y;
			wgt   += w*read_wgt;
			
			read_idx = get_ix_3d(should_conj,p_x  ,p_y  ,p_z+1,M,N);
			read_val = t_val[read_idx];
			read_wgt = t_wgt[read_idx];
			if(should_conj) read_val.y = -read_val.y;
			w = (1-w_x)*(1-w_y)*(  w_z);
			val.x += w*read_val.x;
			val.y += w*read_val.y;
			wgt   += w*read_wgt;
			
			read_idx = get_ix_3d(should_conj,p_x+1,p_y  ,p_z+1,M,N);
			read_val = t_val[read_idx];
			read_wgt = t_wgt[read_idx];
			if(should_conj) read_val.y = -read_val.y;
			w = (  w_x)*(1-w_y)*(  w_z);
			val.x += w*read_val.x;
			val.y += w*read_val.y;
			wgt   += w*read_wgt;
			
			read_idx = get_ix_3d(should_conj,p_x  ,p_y+1,p_z+1,M,N);
			read_val = t_val[read_idx];
			read_wgt = t_wgt[read_idx];
			if(should_conj) read_val.y = -read_val.y;
			w = (1-w_x)*(  w_y)*(  w_z);
			val.x += w*read_val.x;
			val.y += w*read_val.y;
			wgt   += w*read_wgt;
			
			read_idx = get_ix_3d(should_conj,p_x+1,p_y+1,p_z+1,M,N);
			read_val = t_val[read_idx];
			read_wgt = t_wgt[read_idx];
			if(should_conj) read_val.y = -read_val.y;
			w = (  w_x)*(  w_y)*(  w_z);
			val.x += w*read_val.x;
			val.y += w*read_val.y;
			wgt   += w*read_wgt;

            p_val[idx] = val;
            p_wgt[idx] = wgt;
        }
    }

}


}

#endif /// GPU_KERNEL_VOL_H


