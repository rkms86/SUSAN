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
    
#if __CUDA_ARCH__ < 600
__device__ double atomic_Add(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#else
__device__ double atomic_Add(double* address, double val) {
    return atomicAdd(address,val);
}
#endif

__device__ void SN2(float&a,float&b,float&tmp) {
    tmp = a;
    a = max(b,a);
    b = min(b,tmp);
}

__device__ int3 get_th_idx() {
    return make_int3(threadIdx.x + blockIdx.x*blockDim.x,threadIdx.y + blockIdx.y*blockDim.y,threadIdx.z + blockIdx.z*blockDim.z);
}

__device__ bool first_thread_in_block() {
    return (threadIdx.x==0) && (threadIdx.y==0) && (threadIdx.z==0);
}

__device__ long get_2d_idx(const int x,const int y,const int3&ss_siz) {
    return x + y*ss_siz.x;
}

__device__ long get_2d_idx(const int3&ss_idx,const int3&ss_siz) {
    return get_2d_idx(ss_idx.x,ss_idx.y,ss_siz);
}

__device__ long get_3d_idx(const int x,const int y,const int z,const int X,const int Y) {
    return x + y*X + z*X*Y;
}

__device__ long get_3d_idx(const int x,const int y,const int z,const int3&ss_siz) {
    return get_3d_idx(x,y,z,ss_siz.x,ss_siz.y);
}

__device__ long get_3d_idx(const int3&ss_idx,const int3&ss_siz) {
    return get_3d_idx(ss_idx.x,ss_idx.y,ss_idx.z,ss_siz.x,ss_siz.y);
}

__device__ long get_3d_idx(const uint3&ss_idx,const dim3&ss_siz) {
    return get_3d_idx(ss_idx.x,ss_idx.y,ss_idx.z,ss_siz.x,ss_siz.y);
}

__device__ int fftshift_idx(const int idx, const int center) {
    return (idx<center) ? idx + center : idx - center;
}

__device__ float l2_distance(const float x, const float y) {
    return sqrtf( x*x + y*y );
}

__device__ float l2_distance(const float x, const float y, const float z) {
    return sqrtf( x*x + y*y + z*z );
}

__device__ void get_xyR(float&x,float&y,float&R,const float in_x,const float in_y) {
    x = in_x;
    y = in_y;
    R = l2_distance(x,y);
}

__device__ void get_xyR_unit(float&x,float&y,float&R,const float in_x,const float in_y) {
    x = in_x;
    y = in_y;
    R = l2_distance(x,y);
    if( R<0.5 ) R = 1;
    x = x/R;
    y = y/R;
}

__device__ float get_bp_wgt(const float min_R,const float max_R,const float rolloff,const float R) {
    float roll_w = max(rolloff,1.0);
    float ang,wgt_lp,wgt_hp;
    ang = (R-max_R)/roll_w;
    ang = M_PI*min(max(ang,0.0),1.0);
    wgt_lp = 0.5*cosf(ang) + 0.5;

    ang = (min_R-R)/roll_w;
    ang = M_PI*min(max(ang,0.0),1.0);
    wgt_hp = 0.5*cosf(ang) + 0.5;

    return wgt_lp*wgt_hp;
}

/// HOST FUNCTIONS:
__global__ void rotate_180_stk(float*p_out,const float*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long ix_in  = get_3d_idx(ss_idx,ss_siz);
        ss_idx.x    = ss_siz.x - ss_idx.x;
        ss_idx.y    = ss_siz.y - ss_idx.y;
        long ix_out = get_3d_idx(ss_idx,ss_siz);

        float val = p_in[ix_in];
        p_out[ix_out] = val;
    }
}

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

__global__ void fftshift3D(float*p_work, const int N) {

    int3 ss_idx = get_th_idx();

    int center = N/2;

    if( ss_idx.x < center && ss_idx.y < N && ss_idx.z < N ) {

        int3 ss_siz = make_int3(N,N,N);
        long ix_A = get_3d_idx(ss_idx,ss_siz);
        ss_idx.x = ss_idx.x + center;
        ss_idx.y = fftshift_idx(ss_idx.y,center);
        ss_idx.z = fftshift_idx(ss_idx.z,center);
        long ix_B = get_3d_idx(ss_idx,ss_siz);

        float val_A = p_work[ ix_A ];
        float val_B = p_work[ ix_B ];

        p_work[ ix_A ] = val_B;
        p_work[ ix_B ] = val_A;

    }
}

__global__ void fftshift3D(float2*p_work, const int M,const int N) {

    int3 ss_idx = get_th_idx();

    int center = N/2;

    if( ss_idx.x < M && ss_idx.y < N && ss_idx.z < center ) {

        int3 ss_siz = make_int3(M,N,N);
        long ix_A = get_3d_idx(ss_idx,ss_siz);
        ss_idx.y = fftshift_idx(ss_idx.y,center);
        ss_idx.z = fftshift_idx(ss_idx.z,center);
        long ix_B = get_3d_idx(ss_idx,ss_siz);

        float2 val_A = p_work[ ix_A ];
        float2 val_B = p_work[ ix_B ];

        p_work[ ix_A ] = val_B;
        p_work[ ix_B ] = val_A;

    }
}

__global__ void fftshift3D(double*p_work, const int M,const int N) {

    int3 ss_idx = get_th_idx();

    int center = N/2;

    if( ss_idx.x < M && ss_idx.y < N && ss_idx.z < center ) {

        int3 ss_siz = make_int3(M,N,N);
        long ix_A = get_3d_idx(ss_idx,ss_siz);
        ss_idx.y = fftshift_idx(ss_idx.y,center);
        ss_idx.z = fftshift_idx(ss_idx.z,center);
        long ix_B = get_3d_idx(ss_idx,ss_siz);

        double val_A = p_work[ ix_A ];
        double val_B = p_work[ ix_B ];

        p_work[ ix_A ] = val_B;
        p_work[ ix_B ] = val_A;

    }
}

__global__ void load_surf(cudaSurfaceObject_t out_surf,const float*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float v = p_in[ get_3d_idx(ss_idx,ss_siz) ];
        surf2DLayeredwrite<float>(v,out_surf,ss_idx.x*sizeof(float), ss_idx.y, ss_idx.z);
    }
}

__global__ void load_surf_dilate_1(cudaSurfaceObject_t out_surf,const float*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float v = p_in[ get_3d_idx(ss_idx,ss_siz) ];

        if( (ss_idx.x > 0) && (ss_idx.x < (ss_siz.x-1)) ) {
            if( (ss_idx.y > 0) && (ss_idx.y < (ss_siz.y-1)) ) {

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y-1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x  ,ss_idx.y-1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y-1,ss_idx.z,ss_siz) ]);

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y  ,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y  ,ss_idx.z,ss_siz) ]);

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y+1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x  ,ss_idx.y+1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y+1,ss_idx.z,ss_siz) ]);
            }
        }

        surf2DLayeredwrite<float>(v,out_surf,ss_idx.x*sizeof(float), ss_idx.y, ss_idx.z);
    }
}

__global__ void load_surf_dilate_2(cudaSurfaceObject_t out_surf,const float*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float v = p_in[ get_3d_idx(ss_idx,ss_siz) ];

        if( (ss_idx.x > 1) && (ss_idx.x < (ss_siz.x-2)) ) {
            if( (ss_idx.y > 1) && (ss_idx.y < (ss_siz.y-2)) ) {

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y-2,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x  ,ss_idx.y-2,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y-2,ss_idx.z,ss_siz) ]);

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-2,ss_idx.y-1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y-1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x  ,ss_idx.y-1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y-1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+2,ss_idx.y-1,ss_idx.z,ss_siz) ]);

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-2,ss_idx.y  ,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y  ,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y  ,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+2,ss_idx.y  ,ss_idx.z,ss_siz) ]);

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-2,ss_idx.y+1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y+1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x  ,ss_idx.y+1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y+1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+2,ss_idx.y+1,ss_idx.z,ss_siz) ]);

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y+2,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x  ,ss_idx.y+2,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y+2,ss_idx.z,ss_siz) ]);
            }
        }

        surf2DLayeredwrite<float>(v,out_surf,ss_idx.x*sizeof(float), ss_idx.y, ss_idx.z);
    }
}

__global__ void load_surf_dilate_3(cudaSurfaceObject_t out_surf,const float*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float v = p_in[ get_3d_idx(ss_idx,ss_siz) ];

        if( (ss_idx.x > 2) && (ss_idx.x < (ss_siz.x-3)) ) {
            if( (ss_idx.y > 2) && (ss_idx.y < (ss_siz.y-3)) ) {

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y-3,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x  ,ss_idx.y-3,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y-3,ss_idx.z,ss_siz) ]);

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-2,ss_idx.y-2,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y-2,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x  ,ss_idx.y-2,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y-2,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+2,ss_idx.y-2,ss_idx.z,ss_siz) ]);

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-3,ss_idx.y-1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-2,ss_idx.y-1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y-1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x  ,ss_idx.y-1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y-1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+2,ss_idx.y-1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+3,ss_idx.y-1,ss_idx.z,ss_siz) ]);

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-3,ss_idx.y  ,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-2,ss_idx.y  ,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y  ,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y  ,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+2,ss_idx.y  ,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+3,ss_idx.y  ,ss_idx.z,ss_siz) ]);

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-3,ss_idx.y+1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-2,ss_idx.y+1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y+1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x  ,ss_idx.y+1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y+1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+2,ss_idx.y+1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+3,ss_idx.y+1,ss_idx.z,ss_siz) ]);

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-2,ss_idx.y+2,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y+2,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x  ,ss_idx.y+2,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y+2,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+2,ss_idx.y+2,ss_idx.z,ss_siz) ]);

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y+3,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x  ,ss_idx.y+3,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y+3,ss_idx.z,ss_siz) ]);
            }
        }

        surf2DLayeredwrite<float>(v,out_surf,ss_idx.x*sizeof(float), ss_idx.y, ss_idx.z);
    }
}

__global__ void load_surf_3(cudaSurfaceObject_t out_surf,const float*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float v = p_in[ get_3d_idx(ss_idx,ss_siz) ];
        surf3Dwrite<float>(v,out_surf,ss_idx.x*sizeof(float), ss_idx.y, ss_idx.z);
    }
}

__global__ void load_surf_3(cudaSurfaceObject_t out_surf,const float2*p_in,const int3 ss_siz,bool conjugate=false) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 v = p_in[ get_3d_idx(ss_idx,ss_siz) ];
        if( conjugate )
            v.y = -v.y;
        surf3Dwrite<float2>(v,out_surf,ss_idx.x*sizeof(float2), ss_idx.y, ss_idx.z);
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

__global__ void load_surf_real(cudaSurfaceObject_t out_surf,const float2*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 val = p_in[ get_3d_idx(ss_idx,ss_siz) ];
        surf2DLayeredwrite<float>(val.x,out_surf,ss_idx.x*sizeof(float), ss_idx.y, ss_idx.z);
    }
}

__global__ void load_surf_real_positive(cudaSurfaceObject_t out_surf,const float2*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 val = p_in[ get_3d_idx(ss_idx,ss_siz) ];
        float v = max(val.x,0.0f);
        surf2DLayeredwrite<float>(v,out_surf,ss_idx.x*sizeof(float), ss_idx.y, ss_idx.z);
    }
}

__global__ void conv_gaussian(float*p_out,const float*p_in,float num,float scl,const int3 ss_siz) {
    ///    num    scl:
    /// 0.5000, 6.2831
    /// 0.2500,12.5372
    /// 0.1250,23.9907

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float val = 0;

        for(int y=-4;y<5;y++) {
            for(int x=-4;x<5;x++) {
                float r = l2_distance(x,y);
                float wgt = exp(-num*r*r)/scl;
                int X = ss_idx.x + x;
                int Y = ss_idx.y + y;
                if( X < 0 ) {
                    X = -X;
                    Y = ss_siz.y-Y;
                }
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

__global__ void stk_medfilt(float*p_out,const float*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float v0,v1,v2,v3,v4,v5,v6,v7,v8,tmp;
        int y, x;

        /// unrolled to avoid using local arrays

        /// Y = -1
        y = ss_idx.y - 1;
        if( y < 0 ) y = -y;

        x = ss_idx.x - 1;
        if( x < 0 ) x = -x;
        v0 = p_in[ get_3d_idx(x,y,ss_idx.z,ss_siz) ];

        x = ss_idx.x;
        v1 = p_in[ get_3d_idx(x,y,ss_idx.z,ss_siz) ];

        x = ss_idx.x + 1;
        if( x >= ss_siz.x ) x = 2*(ss_siz.x-1)-x;
        v2 = p_in[ get_3d_idx(x,y,ss_idx.z,ss_siz) ];

        /// Y = 0
        y = ss_idx.y;

        x = ss_idx.x - 1;
        if( x < 0 ) x = -x;
        v3 = p_in[ get_3d_idx(x,y,ss_idx.z,ss_siz) ];

        x = ss_idx.x;
        v4 = p_in[ get_3d_idx(x,y,ss_idx.z,ss_siz) ];

        x = ss_idx.x + 1;
        if( x >= ss_siz.x ) x = 2*(ss_siz.x-1)-x;
        v5 = p_in[ get_3d_idx(x,y,ss_idx.z,ss_siz) ];

        /// Y = +1
        y = ss_idx.y + 1;
        if( y >= ss_siz.y ) y = 2*(ss_siz.y-1)-y;

        x = ss_idx.x - 1;
        if( x < 0 ) x = -x;
        v6 = p_in[ get_3d_idx(x,y,ss_idx.z,ss_siz) ];

        x = ss_idx.x;
        v7 = p_in[ get_3d_idx(x,y,ss_idx.z,ss_siz) ];

        x = ss_idx.x + 1;
        if( x >= ss_siz.x ) x = 2*(ss_siz.x-1)-x;
        v8 = p_in[ get_3d_idx(x,y,ss_idx.z,ss_siz) ];

        /// SORT!
        SN2(v0,v1,tmp);
        SN2(v2,v3,tmp);
        SN2(v4,v5,tmp);
        SN2(v6,v7,tmp);
        SN2(v0,v2,tmp);
        SN2(v4,v6,tmp);
        SN2(v1,v3,tmp);
        SN2(v5,v7,tmp);
        SN2(v1,v2,tmp);
        SN2(v6,v7,tmp);
        SN2(v0,v4,tmp);
        SN2(v1,v5,tmp);
        SN2(v2,v6,tmp);
        SN2(v3,v7,tmp);
        SN2(v2,v4,tmp);
        SN2(v3,v5,tmp);
        SN2(v1,v2,tmp);
        SN2(v3,v4,tmp);
        SN2(v5,v6,tmp);
        SN2(v0,v7,tmp);
        SN2(v4,v8,tmp);
        SN2(v2,v4,tmp);

        p_out[get_3d_idx(ss_idx,ss_siz)] = v4;
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

__global__ void get_std_from_fourier_stk(float*p_std,const float2*p_data,const float3 bandpass,const int3 ss_siz) {

    __shared__ float local_std[1];
    if( first_thread_in_block() )
        local_std[0] = 0;
    __syncthreads();

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float R = l2_distance(ss_idx.x,ss_idx.y - ss_siz.y/2);
        float bp = get_bp_wgt(bandpass.x,bandpass.y,bandpass.z,R);

        if( (bp > 0.05) && (R > 0.5) ) {
            long idx = get_3d_idx(ss_idx,ss_siz);
            float2 val = p_data[idx];
            val.x *= bp;
            val.y *= bp;
            float acc = cuCabsf(val);
            acc = acc*acc;
            if( ss_idx.x > 0 )
                acc *= 2;
            atomicAdd( local_std , acc );
        }
        __syncthreads();

        if(first_thread_in_block()) {
            float acc = local_std[0];
            atomicAdd( p_std+ss_idx.z , acc );
        }
    }
}

__global__ void apply_std_to_fourier_stk(float2*p_data,const float*p_std,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long idx = get_3d_idx(ss_idx,ss_siz);
        float2 val = p_data[idx];
        float wgt = p_std[ss_idx.z];
        wgt   = sqrtf(wgt);
        val.x = val.x/wgt;
        val.y = val.y/wgt;
        p_data[idx] = val;
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

__global__ void apply_circular_mask(float*p_out,const float*p_in,const float2*p_bp,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long idx = get_3d_idx(ss_idx,ss_siz);
        float val = 0;

        float R = l2_distance(ss_idx.x-ss_siz.x/2,ss_idx.y-ss_siz.y/2);

        float w = get_bp_wgt(p_bp[ss_idx.z].x,p_bp[ss_idx.z].y,10,R);

        if( w > 0.05 ) {
            val = w*p_in[ idx ];
        }
        p_out[idx] = val;
    }
}

__global__ void stk_scale(float*p_data,const float2*p_bp,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long idx = get_3d_idx(ss_idx,ss_siz);
        float val = p_data[idx];
        float w = p_bp[ss_idx.z].y;
        w = M_PI*w*w;
        val = p_data[ idx ]/w;
        p_data[idx] = val;
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

__global__ void radial_ps_avg(float*p_avg,float*p_wgt,const float2*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 val = p_in[ get_3d_idx(ss_idx,ss_siz) ];
        float  x = ss_idx.x;
        float  y = ss_idx.y-ss_siz.y/2;
        float  R = l2_distance(x,y);
        int    r = (int)roundf(R);
        int  idx = r + ss_siz.x*ss_idx.z;
        float out = cuCabsf(val);
        if( r < ss_siz.x ) {
            atomicAdd(p_avg + idx,out);
            atomicAdd(p_wgt + idx,1.0);
        }
    }
}

__global__ void radial_ps_avg_double_side(float*p_avg,float*p_wgt,const float*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float val = p_in[ get_3d_idx(ss_idx,ss_siz) ];
        float  x = ss_idx.x;
        float  y = ss_idx.y-ss_siz.y/2;
        float  R = l2_distance(x,y);
        int    r = (int)roundf(R);
        int  idx = r + ss_siz.y*ss_idx.z;
        if( r < ss_siz.x ) {
            atomicAdd(p_avg + idx,val);
            atomicAdd(p_wgt + idx,1.0);

            r = ss_siz.y - r;
            if( r < ss_siz.y ) {
                idx = r + ss_siz.y*ss_idx.z;
                atomicAdd(p_avg + idx,val);
                atomicAdd(p_wgt + idx,1.0);
            }
        }

    }
}

__global__ void radial_ps_norm(float2*p_data,const float*p_avg,const float*p_wgt,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 val = {0,0};
        float  x = ss_idx.x;
        float  y = ss_idx.y-ss_siz.y/2;
        float  R = l2_distance(x,y);
        int    r = (int)roundf(R);

        if( r < ss_siz.x ) {
            val = p_data[ get_3d_idx(ss_idx,ss_siz) ];
            int  idx = r + ss_siz.x*ss_idx.z;
            float avg = p_avg[idx];
            float wgt = p_wgt[idx];

            wgt = max(wgt,1.0);
            avg = avg/wgt;

            if( avg > 1e-7 ) {
                val = p_data[ get_3d_idx(ss_idx,ss_siz) ];
                val.x = val.x/avg;
                val.y = val.y/avg;
            }
            p_data[ get_3d_idx(ss_idx,ss_siz) ] = val;
        }
    }
}

__global__ void radial_frc_avg(float*p_avg,float*p_wgt,const float2*p_in,const float3 bandpass,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 val = p_in[ get_3d_idx(ss_idx,ss_siz) ];
        float  x = ss_idx.x;
        float  y = ss_idx.y-(ss_siz.y/2);
        float  R = l2_distance(x,y);
        float bp = get_bp_wgt(bandpass.x,bandpass.y,bandpass.z,R);
        int    r = (int)roundf(R);
        int  idx = r + ss_siz.x*ss_idx.z;
        float out = (val.x*val.x) + (val.y*val.y);
        if( (r < ss_siz.x) && (bp > 0.02) && (cuCabsf(val)>1e-8) ) {
            atomicAdd(p_avg + idx,out);
            atomicAdd(p_wgt + idx,1.0);
        }
    }
}

__global__ void radial_frc_norm(float2*p_data,const float*p_avg,const float*p_wgt,const float ssnr_F,const float ssnr_S,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 val = {0,0};
        float  x = ss_idx.x;
        float  y = ss_idx.y-ss_siz.y/2;
        float  R = l2_distance(x,y);
        int    r = (int)roundf(R);

        if( r < ss_siz.x ) {
            int    idx;
            double avg;
            double wgt;
            double w_acc = 0;

            for(int i=1;i<ss_siz.x;i++) {
                idx = r + ss_siz.x*ss_idx.z;
                avg = p_avg[idx];
                wgt = p_wgt[idx];
                w_acc += (avg / max(wgt,1.0));
            }
            w_acc = sqrt(w_acc);

            idx = r + ss_siz.x*ss_idx.z;
            avg = p_avg[idx];
            wgt = p_wgt[idx];

            if( avg > 0 ) {

                wgt = max(wgt,1.0);
                avg = avg/wgt;
                avg = sqrt(avg)/w_acc;

                if( avg > 1e-10 ) {
                    val = p_data[ get_3d_idx(ss_idx,ss_siz) ];
                    if( r > 0 ) {
                        float w_avg = avg;
                        if(ssnr_S>1)
                            w_avg += (1/(ssnr_S*exp(R*ssnr_F)));
                        val.x = val.x/w_avg;
                        val.y = val.y/w_avg;
                    }
                }
            }
        }

        p_data[ get_3d_idx(ss_idx,ss_siz) ] = val;
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

__global__ void divide(cudaSurfaceObject_t surf,const float wgt,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 val = surf3Dread<float2>(surf,ss_idx.x*sizeof(float2), ss_idx.y, ss_idx.z);
        val.x = val.x/wgt;
        val.y = val.y/wgt;
        surf3Dwrite<float2>(val,surf,ss_idx.x*sizeof(float2), ss_idx.y, ss_idx.z);
    }
}

__global__ void divide(float2*p_avg,const float wgt,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        int idx = get_3d_idx(ss_idx,ss_siz);
        float2 val = p_avg[ idx ];
        val.x = val.x/wgt;
        val.y = val.y/wgt;
        p_avg[ idx ] = val;
    }
}

__global__ void divide(float*p_avg,const float wgt,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        int idx = get_3d_idx(ss_idx,ss_siz);
        float val = p_avg[ idx ];
        val = val/wgt;
        p_avg[ idx ] = val;
    }
}

__global__ void load_pad(float*p_out,const float*p_in,const int3 half_pad,const int3 ss_raw,const int3 ss_pad) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_raw.x && ss_idx.y < ss_raw.y && ss_idx.z < ss_raw.z ) {

        int idx_in  = get_3d_idx(ss_idx,ss_raw);
        int idx_out = get_3d_idx(ss_idx.x+half_pad.x,ss_idx.y+half_pad.y,ss_idx.z+half_pad.z,ss_pad);
        float data = p_in[idx_in];
        p_out[idx_out] = data;
    }

}

__global__ void remove_pad_vol(float*p_out,const float*p_in,const int half_pad,const int3 ss_raw,const int3 ss_pad) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_raw.x && ss_idx.y < ss_raw.y && ss_idx.z < ss_raw.z ) {

        long idx_in  = get_3d_idx(ss_idx.x+half_pad,ss_idx.y+half_pad,ss_idx.z+half_pad,ss_pad);
        long idx_out = get_3d_idx(ss_idx,ss_raw);
        float data = p_in[idx_in];
        p_out[idx_out] = data;
    }

}

__global__ void remove_pad_stk(float*p_out,const float*p_in,const int half_pad,const int3 ss_raw,const int3 ss_pad) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_raw.x && ss_idx.y < ss_raw.y && ss_idx.z < ss_raw.z ) {

        long idx_in  = get_3d_idx(ss_idx.x+half_pad,ss_idx.y+half_pad,ss_idx.z,ss_pad);
        long idx_out = get_3d_idx(ss_idx,ss_raw);
        float data = p_in[idx_in];
        p_out[idx_out] = data;
    }

}

__global__ void subpixel_shift(float2*p_data,const Proj2D*g_ali,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        int idx  = get_3d_idx(ss_idx,ss_siz);

        float x,y,R;
        get_xyR_unit(x,y,R,ss_idx.x,ss_idx.y-ss_siz.y/2);

        float2 data = p_data[idx];

        float phase_shift = x*g_ali[ss_idx.z].t.x + y*g_ali[ss_idx.z].t.y;
        float sin_cos_arg = -2*M_PI*phase_shift/ss_siz.y;

        /// exp(-ix) = cos(x) - i sin(x);
        float tmp_real = data.x;
        float tmp_imag = data.y;
        float c = cos(sin_cos_arg);
        float s = sin(sin_cos_arg);
        data.x = tmp_real*c + tmp_imag*s;
        data.y = tmp_imag*c - tmp_real*s;

        p_data[idx] = data;
    }

}

__global__ void multiply(float2*p_out,const double2*p_acc,const double*p_wgt,const int3 ss_siz,const double scale=1.0) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {
        int idx = get_3d_idx(ss_idx,ss_siz);
        double2 val = p_acc[ idx ];
        double  wgt = p_wgt[ idx ];
        float2 rslt;
        rslt.x = (float)(val.x*wgt*scale);
        rslt.y = (float)(val.y*wgt*scale);
        p_out[ idx ] = rslt;
    }
}

__global__ void multiply(float2*p_out,const float2*p_in,const float*p_wgt,const int3 ss_siz,const double scale=1.0) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {
        int idx = get_3d_idx(ss_idx,ss_siz);
        float2 val = p_in [ idx ];
        float  wgt = p_wgt[ idx ];
        float2 rslt;
        rslt.x = (float)(val.x*wgt*scale);
        rslt.y = (float)(val.y*wgt*scale);
        p_out[ idx ] = rslt;
    }
}

__global__ void multiply(float2*p_out,const float*p_wgt,const int3 ss_siz,const float scale=1.0) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {
        int idx = get_3d_idx(ss_idx,ss_siz);
        float2 val = p_out[ idx ];
        float  wgt = p_wgt[ idx ];
        float2 rslt;
        rslt.x = (val.x*wgt*scale);
        rslt.y = (val.y*wgt*scale);
        p_out[ idx ] = rslt;
    }
}

__global__ void multiply(float2*p_out,const float2*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {
        int idx = get_3d_idx(ss_idx,ss_siz);
        float2 val = p_out[ idx ];
        float2 wgt = p_in [ idx ];
        float2 rslt;
        rslt = cuCmulf(val,wgt);
        p_out[ idx ] = rslt;
    }
}

__global__ void print_proj2D(Proj2D*g_tlt,const int in_K) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x == 0 && ss_idx.y == 0 && ss_idx.z == 0 ) {
        for( int z=0; z<in_K; z++ ) {
            //if( (z == 11) || (z == 21) || (z == 31) ) {
            if( z == 20 ) {
                printf("R[%d,0,:] = (%f,%f,%f)\n",z,g_tlt[z].R.xx,g_tlt[z].R.xy,g_tlt[z].R.xz);
                printf("R[%d,1,:] = (%f,%f,%f)\n",z,g_tlt[z].R.yx,g_tlt[z].R.yy,g_tlt[z].R.yz);
                printf("R[%d,2,:] = (%f,%f,%f)\n",z,g_tlt[z].R.zx,g_tlt[z].R.zy,g_tlt[z].R.zz);
            }
        }
    }
}

__global__ void rotate_post(Proj2D*g_tlt,Rot33 R,Proj2D*g_tlt_in,const int in_K) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < in_K*9 && ss_idx.y == 0 && ss_idx.z == 0 ) {

        int mod_z = ss_idx.x%9;
        int z = (ss_idx.x - mod_z)/9;
        int x = mod_z % 3;
        int y = (mod_z - x)/3;

        float *pR = &(R.xx);
        float *r_out = &(g_tlt[z].R.xx);
        float *r_in  = &(g_tlt_in[z].R.xx);

        float rslt = 0;

        /// rslt = pR * r_in
        /// rslt = R * g_tlt[z].R
        rslt += pR[y*3  ]*r_in[x  ];
        rslt += pR[y*3+1]*r_in[x+3];
        rslt += pR[y*3+2]*r_in[x+6];

        r_out[mod_z] = rslt;
    }

    __syncthreads();

    if( ss_idx.x < in_K && ss_idx.y == 0 && ss_idx.z == 0 ) {
        g_tlt[ss_idx.x].w = g_tlt_in[ss_idx.x].w;
    }

}

__global__ void rotate_pre(Proj2D*g_tlt,Rot33 R,Proj2D*g_tlt_in,const int in_K) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < in_K*9 && ss_idx.y == 0 && ss_idx.z == 0 ) {

        int mod_z = ss_idx.x%9;
        int z = (ss_idx.x - mod_z)/9;
        int x = mod_z % 3;
        int y = (mod_z - x)/3;

        float *pR = &(R.xx);
        float *r_out = &(g_tlt[z].R.xx);
        float *r_in  = &(g_tlt_in[z].R.xx);

        float rslt = 0;

        /// rslt = r_in * pR
        /// rslt = g_tlt[z].R * R
        rslt += r_in[y*3  ]*pR[x  ];
        rslt += r_in[y*3+1]*pR[x+3];
        rslt += r_in[y*3+2]*pR[x+6];

        r_out[mod_z] = rslt;
    }

    __syncthreads();

    if( ss_idx.x < in_K && ss_idx.y == 0 && ss_idx.z == 0 ) {
        g_tlt[ss_idx.x].w = g_tlt_in[ss_idx.x].w;
    }

}

__global__ void apply_radial_wgt(float2*p_data,const float w_total,float crowther_limit,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        int idx = get_3d_idx(ss_idx,ss_siz);
        float2 val = p_data[ idx ];

        float w_off = 1/w_total;
        float w = ss_idx.x;
        w = fminf(w/crowther_limit,1.0);
        w = (1-w_off)*w + w_off;
        if( (ss_idx.x == 0) || (ss_idx.x+1 == ss_siz.x) )
            w /= 2;
        val.x = w*val.x;
        val.y = w*val.y;

        p_data[ idx ] = val;
    }
}

__global__ void apply_bandpass_fourier(float2*p_w,const float3 bandpass,const int M, const int N, const int K)
{
    int3 ss_idx = get_th_idx();

    if( ss_idx.x < M && ss_idx.y < N && ss_idx.z < K ) {

        long idx = ss_idx.x + M*ss_idx.y + M*N*ss_idx.z;

        float2 val = {0,0};

        float R = l2_distance(ss_idx.x,ss_idx.y - N/2);
        float bp = get_bp_wgt(bandpass.x,bandpass.y,bandpass.z,R);

        if( bp > 0.025 ) {
            val = p_w[ idx ];
            val.x *= bp;
            val.y *= bp;
        }

        p_w[ idx ] = val;
    }
}



__global__ void norm_complex(float2*p_data,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        int idx = get_3d_idx(ss_idx,ss_siz);
        float2 val = p_data[ idx ];

        float wgt = l2_distance(val.x,val.y);
        if( wgt < 0.000001 )
            wgt = 1;

        val.x = val.x*wgt;
        val.y = val.y*wgt;

        p_data[ idx ] = val;
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
}*/

}

#endif /// GPU_KERNEL_H


