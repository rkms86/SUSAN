#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cufft.h"

#include "gpu.h"

namespace GpuKernels {

/// DEVICE FUNCTIONS:

__device__ int3 get_th_idx() {
	return make_int3(threadIdx.x + blockIdx.x*blockDim.x,threadIdx.y + blockIdx.y*blockDim.y,threadIdx.z + blockIdx.z*blockDim.z);
}

__device__ long get_2d_idx(const int x,const int y,const int3&ss_siz) {
	return x + y*ss_siz.x;
}

__device__ long get_2d_idx(const int3&ss_idx,const int3&ss_siz) {
	return get_2d_idx(ss_idx.x,ss_idx.y,ss_siz);
}

__device__ long get_3d_idx(const int x,const int y,const int z,const int3&ss_siz) {
	return x + y*ss_siz.x + z*ss_siz.x*ss_siz.y;
}

__device__ long get_3d_idx(const int3&ss_idx,const int3&ss_siz) {
	return get_3d_idx(ss_idx.x,ss_idx.y,ss_idx.z,ss_siz);
}

__device__ int fftshift_idx(const int idx, const int center) {
    return (idx<center) ? idx + center : idx - center;
}

__device__ float l2_distance(const float x, const float y) {
    return sqrtf( x*x + y*y );
}

/// HOST FUNCTIONS:
__global__ void fftshift2D(float*p_work,const int3 ss_siz) {

	int3 ss_idx = get_th_idx();

    int Xh = ss_siz.x/2;
    int Yh = ss_siz.y/2;

    if( ss_idx.x < Xh && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long ix_A = get_3d_idx(ss_idx,ss_siz);
        ss_idx.x = ss_idx.x + Xh;
        ss_idx.y = fftshift_idx(ss_idx.y,Yh);
        long ix_B = get_3d_idx(ss_idx,ss_siz);

        float in_A = p_work[ix_A];
        float in_B = p_work[ix_B];

        p_work[ix_A] = in_B;
        p_work[ix_B] = in_A;
    }
}

__global__ void fftshift2D(float2*p_work,const int3 ss_siz) {

	int3 ss_idx = get_th_idx();

    int Yh = ss_siz.y/2;

    if( ss_idx.x < ss_siz.x && ss_idx.y < Yh && ss_idx.z < ss_siz.z ) {

        long ix_A = get_3d_idx(ss_idx,ss_siz);
        ss_idx.y = fftshift_idx(ss_idx.y,Yh);
        long ix_B = get_3d_idx(ss_idx,ss_siz);

        float2 in_A = p_work[ix_A];
        float2 in_B = p_work[ix_B];

        p_work[ix_A] = in_B;
        p_work[ix_B] = in_A;
    }
}

__global__ void load_surf(cudaSurfaceObject_t out_surf,const float*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float v = p_in[ get_3d_idx(ss_idx,ss_siz) ];
        surf2DLayeredwrite<float>(v,out_surf,ss_idx.x*sizeof(float), ss_idx.y, ss_idx.z);
    }
}

__global__ void load_surf_abs(cudaSurfaceObject_t out_surf,const float2*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 val = p_in[ get_3d_idx(ss_idx,ss_siz) ];
        float v = cuCabsf(val);
        surf2DLayeredwrite<float>(v,out_surf,ss_idx.x*sizeof(float), ss_idx.y, ss_idx.z);
    }
}

__global__ void conv_gaussian(float*p_out,const float*p_in,const int3 ss_siz) {
	
	int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

		float num = -0.5774;
		float scl = 5.3885;

		float val = 0;
		
		for(int y=-2;y<3;y++) {
			for(int x=-2;x<3;x++) {
				float r = l2_distance(x,y);
				float wgt = exp(num*r*r)/scl;
				int X = ss_idx.x + x;
				int Y = ss_idx.y + y;
				if( X < 0 )
					X = -X;
				if( Y < 0 )
					Y = -Y;
				if( X >= ss_siz.x )
					X = 2*(ss_siz.x-1)-X;
				if( Y >= ss_siz.y )
					Y = 2*(ss_siz.y-1)-Y;
				val += wgt*p_in[ get_3d_idx(X,Y,ss_idx.z,ss_siz) ];
			}
		}
		
        p_out[get_3d_idx(ss_idx,ss_siz)] = val;
    }
}

__global__ void get_avg_std(float*p_std, float*p_avg, const float*p_in, const int3 ss_siz) {
    
    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

		float N = ss_siz.x*ss_siz.y;
		long idx = get_3d_idx(ss_idx,ss_siz);
		float val = p_in[idx];
		float avg = val/N;
		float std = val*avg;
		atomicAdd(p_avg+ss_idx.z,avg);
		atomicAdd(p_std+ss_idx.z,std);
    }
}

__global__ void zero_avg_one_std(float*p_in, const float*p_std, const float*p_avg, const int3 ss_siz) {
    
    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

		float avg = p_avg[ss_idx.z];
		float std = sqrtf( p_std[ss_idx.z] - p_avg[ss_idx.z]*p_avg[ss_idx.z] );

		long idx = get_3d_idx(ss_idx,ss_siz);
		float val = p_in[idx];
		p_in[idx] = (val-avg)/std;
    }
}

__global__ void expand_ps_hermitian(float*p_out,const float*p_in,const int3 ss_siz) {
	
	int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

		int Nh = ss_siz.y/2;
		int x = ss_idx.x - Nh;
		int y = ss_idx.y - Nh;
		
		if( x<0 ) {
			x = -x;
			y = -y;
		}
		
		x = x+Nh;
		y = y+Nh;
		
		if(x<0) x = Nh+x;
		if(y<0) y = ss_siz.y+y;
		
		if(x>=Nh) x = x-Nh;
		if(y>=ss_siz.y) y = y-ss_siz.y;
		
		float val = p_in[ x + y*(Nh+1) + ss_idx.z*ss_siz.y*(Nh+1) ];
		
        p_out[get_3d_idx(ss_idx,ss_siz)] = val;
    }
}

__global__ void load_abs(float*p_out,const float2*p_in,const int3 ss_siz) {
	
	int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

		long idx = get_3d_idx(ss_idx,ss_siz);
		float2 val = p_in[idx];
        float v = cuCabsf(val);
        p_out[idx] = v;
    }
}

__global__ void radial_ps_avg(float*p_avg,float*p_wgt,const float*p_in,const int3 ss_siz) {
	
	int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

		float val = p_in[ get_3d_idx(ss_idx,ss_siz) ];
		float x = ss_idx.x;
		float y = ss_idx.y-ss_siz.y/2;
		float R = l2_distance(x,y);
		int   r = (int)roundf(R);
		int idx = r + ss_siz.x*ss_idx.z;
		if( r < ss_siz.x ) {
			atomicAdd(p_avg + idx,val);
			atomicAdd(p_wgt + idx,1.0);
		}
    }
}

__global__ void divide(float*p_avg,const float*p_wgt,const int3 ss_siz) {
	
	int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

		int idx = get_3d_idx(ss_idx,ss_siz);
		float val = p_avg[ idx ];
		float wgt = p_wgt[ idx ];
		if( wgt < 1 ) wgt = 1;
		p_avg[ idx ] = val/wgt;
    }
}


/*
__global__ void fftshift_and_load_surf(cudaSurfaceObject_t out_surf, const float*p_in, const int N, const int K)
{
    int thx, thy, thz;
    get_th_idx(thx,thy,thz);

    if( thx < N && thy < N && thz < K ) {
        float val = p_in[ get_3d_idx(thx,thy,thz,N,N) ];
        int x = fftshift_idx(thx,N/2);
        int y = fftshift_idx(thy,N/2);
        surf2DLayeredwrite<float>(val,out_surf,x*sizeof(float),y,thz);
    }
}

__global__ void rotate_post( Proj2D*g_tlt, Rot33 R, Proj2D*g_tlt_in, const int in_K ) {

    int thx = threadIdx.x + blockIdx.x*blockDim.x;
    int thy = threadIdx.y + blockIdx.y*blockDim.y;
    int thz = threadIdx.z + blockIdx.z*blockDim.z;

    if( thx < in_K*9 && thy == 0 && thz == 0 ) {

        int mod_z = thx%9;
        int z = (thx - mod_z)/9;
        int x = mod_z % 3;
        int y = (mod_z - x)/3;

        float *pR = &(R.xx);
        float *r_out = &(g_tlt[z].R.xx);
        float *r_in  = &(g_tlt_in[z].R.xx);

        float rslt = 0;

        rslt += pR[y*3  ]*r_in[x  ];
        rslt += pR[y*3+1]*r_in[x+3];
        rslt += pR[y*3+2]*r_in[x+6];

        r_out[mod_z] = rslt;
    }

    __syncthreads();

    if( thx < in_K && thy == 0 && thz == 0 ) {
        g_tlt[thx].w = g_tlt_in[thx].w;
    }

}

__global__ void rotate_pre( Proj2D*g_tlt, Rot33 R, Proj2D*g_tlt_in, const int in_K ) {

    int thx = threadIdx.x + blockIdx.x*blockDim.x;
    int thy = threadIdx.y + blockIdx.y*blockDim.y;
    int thz = threadIdx.z + blockIdx.z*blockDim.z;

    if( thx < in_K*9 && thy == 0 && thz == 0 ) {

        int mod_z = thx%9;
        int z = (thx - mod_z)/9;
        int x = mod_z % 3;
        int y = (mod_z - x)/3;

        float *pR = &(R.xx);
        float *r_out = &(g_tlt[z].R.xx);
        float *r_in  = &(g_tlt_in[z].R.xx);

        float rslt = 0;

        rslt += r_in[y*3  ]*pR[x*3  ];
        rslt += r_in[y*3+1]*pR[x*3+1];
        rslt += r_in[y*3+2]*pR[x*3+2];

        r_out[mod_z] = rslt;
    }

    __syncthreads();

    if( thx < in_K && thy == 0 && thz == 0 ) {
        g_tlt[thx].w = g_tlt_in[thx].w;
    }

}*/

}

#endif /// GPU_KERNEL_H

