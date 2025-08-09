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

#ifndef GPU_KERNEL_CTF_H
#define GPU_KERNEL_CTF_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cufft.h"

#include "gpu.h"
#include "gpu_kernel.h"

using namespace GpuKernels;

namespace GpuKernelsCtf {

__device__ float calc_s(const float r, const float R, const float apix) {
    return (r/(R*apix));
}

__device__ float calc_def(const float x, const float y, const float u, const float v, const float ang) {
    float rad = M_PI*ang/180.0;
    float rx = x*cos(rad)-y*sin(rad);
    float ry = x*sin(rad)+y*cos(rad);
    float wx = u*rx;
    float wy = v*ry;
    return l2_distance(wx,wy);
}

__device__ float calc_def(const float x, const float y, const Defocus&def) {
    return calc_def(x,y,def.U,def.V,def.angle);
}

__device__ float calc_gamma(const float def,const float lambda_pi,const float lambda3_Cs_pi_2,const float s2, const float phase_rad=0.0) {
    return (lambda_pi*def*s2 - lambda3_Cs_pi_2*s2*s2 + phase_rad);
}

__device__ float calc_ctf(const float gamma,const float ac,const float ca) {
    return (ca*sin(gamma) + ac*cos(gamma));
}

__device__ float calc_ctf(const float gamma,const float ac) {
    return calc_ctf(gamma,ac,sqrt(1-ac*ac));
}

__device__ float calc_bfactor(const float s,const float bfactor) {
    return expf(-s*s*bfactor/4);
}

__device__ float calc_ssnr(const float r,const float ssnr_F,const float ssnr_S) {
    return (1/(ssnr_S*exp(r*ssnr_F)));
}

__device__ void store_surface(cudaSurfaceObject_t surf,const float val, const int3 pos) {
    surf2DLayeredwrite<float>(val,surf,pos.x*sizeof(float), pos.y, pos.z);
}

__device__ void store_surface(cudaSurfaceObject_t surf,const float2 val, const int3 pos) {
    surf2DLayeredwrite<float2>(val,surf,pos.x*sizeof(float2), pos.y, pos.z);
}

/////////////////

__global__ void create_vec_r(float3*p_vec_r,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < 1 ) {

        float3 rslt;
        rslt.x = ss_idx.x;
        rslt.y = ss_idx.y-ss_siz.y/2;
        rslt.z = l2_distance(rslt.x,rslt.y);
        if( rslt.z > 0 ) {
            rslt.x = rslt.x/rslt.z;
            rslt.y = rslt.y/rslt.z;
        }
        p_vec_r[ get_2d_idx(ss_idx,ss_siz) ] = rslt;
    }
}

__global__ void dwgt_center_ps(float2*p_out,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float x = ss_idx.x;
        float y = ss_idx.y-ss_siz.y/2;
        float R = l2_distance(x,y);
        if(R<16) {
            float2 val = p_out[get_3d_idx(ss_idx,ss_siz)];
            R = 0.7*(7-R);
            float wgt = 1/(1+exp(R));
            val.x *= wgt;
            val.y *= wgt;
            p_out[get_3d_idx(ss_idx,ss_siz)] = val;
        }
    }
}

__global__ void ctf_bin( float*p_out, cudaTextureObject_t texture,
                         const float bin_factor, const int3 ss_siz)
{

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float x = ss_idx.x;
        float y = ss_idx.y;

        if( bin_factor>1 ) {
            x = x/bin_factor;
            float Yh = ss_siz.y/2;
            y = (y-Yh)/bin_factor + Yh;
        }
        y = min(max(y,(float)0.0f),(float)ss_siz.y-1);
        x = min(max(x,(float)0.0f),(float)ss_siz.x-1);

        float val = tex2DLayered<float>(texture,x+0.5,y+0.5,ss_idx.z);
        p_out[ get_3d_idx(ss_idx,ss_siz) ] = val;
    }
}

__global__ void ctf_normalize( float*p_out, cudaTextureObject_t texture,
                               const float2*p_factor, const float3*p_vec_r,
                               const float bin_factor, const int3 ss_siz)
{

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float3 vec_r = p_vec_r[get_2d_idx(ss_idx,ss_siz)];
        /// factor = sqrt(def/(def+delta))
        float factor = vec_r.z*p_factor[ss_idx.z].x;
        float x = factor*vec_r.x;
        float y = factor*vec_r.y + ss_siz.y/2;

        if( bin_factor>1 ) {
            x = x/bin_factor;
            float Yh = ss_siz.y/2;
            y = (y-Yh)/bin_factor + Yh;
        }
        y = min(max(y,(float)0.0f),(float)ss_siz.y-1);
        x = min(max(x,(float)0.0f),(float)ss_siz.x-1);

        float val = tex2DLayered<float>(texture,x+0.5,y+0.5,ss_idx.z);
        p_out[ get_3d_idx(ss_idx,ss_siz) ] = val;
    }
}

__global__ void ctf_radial_normalize( float*p_out, cudaTextureObject_t texture, const float4*p_defocus, const float ix2def,
                                      const float pi_lambda, const float apix, const int3 ss_siz)
{

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float3 vec_r;
        vec_r.x = ss_idx.x;
        vec_r.y = ss_idx.y-ss_siz.y/2;
        vec_r.z = l2_distance(vec_r.x,vec_r.y);
        if( vec_r.z > 0 ) {
            vec_r.x = vec_r.x/vec_r.z;
            vec_r.y = vec_r.y/vec_r.z;
        }

        float s2 = calc_s(vec_r.z,ss_siz.y,apix);
        s2 = s2*s2;
        float def_avg = (p_defocus[ss_idx.z].x+p_defocus[ss_idx.z].y)/2;
        float def_dif = (p_defocus[ss_idx.z].x-p_defocus[ss_idx.z].y);
        float def = calc_def(vec_r.x,vec_r.y,def_dif,0,p_defocus[ss_idx.z].z);
        def = ix2def*def;
        if(def_dif<0) def = -def;
        float factor = vec_r.z*sqrtf( def_avg/(def_avg+def_dif) );
        //float factor = pi_lambda*def*s2;
        //float x = ss_idx.x-factor*vec_r.x;
        //float y = ss_idx.y-factor*vec_r.y;
        //if( vec_r.z-factor >= ss_siz.y/2 ) {
        //    x = ss_idx.x;
        //    y = ss_idx.y;
        //}
        float x = factor*vec_r.x;
        float y = factor*vec_r.y + ss_siz.y/2;

        y = min(max(y,(float)0.0f),(float)ss_siz.y-1);
        x = min(max(x,(float)0.0f),(float)ss_siz.x-1);

        float val = tex2DLayered<float>(texture,x+0.5,y+0.5,ss_idx.z);
        p_out[ get_3d_idx(ss_idx,ss_siz) ] = val;
    }
}

__global__ void accumulate( double*p_acc, double*p_wgt, const float*p_in, const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long idx = get_3d_idx(ss_idx,ss_siz);
        float val = p_in[idx];
        p_acc[ idx ] += val;
        if( abs(val) > SUSAN_FLOAT_TOL ) {
            p_wgt[ idx ]++;
        }

    }
}

__global__ void divide(float*p_out,const double*p_acc,const double*p_wgt,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long ix = get_3d_idx(ss_idx,ss_siz);
        double acc = p_acc[ix];
        double wgt = p_wgt[ix];

        if( wgt < 0.000001 ) wgt = 1;

        p_out[ix] = (float)(acc/wgt);
    }
}

__global__ void rmv_bg(float*p_out, const float*p_in, const float3*p_flt, const int n_flt, const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float val = p_in[ get_3d_idx(ss_idx,ss_siz) ];
        float conv = 0;

        for(int i=0; i<n_flt; i++ ) {
            int x = ss_idx.x + (int)(p_flt[i].x);
            int y = ss_idx.y + (int)(p_flt[i].y);
            float w = p_flt[i].z;

            if( x < 0 ) {x = -x; y = ss_siz.y - y;}
            if( y < 0 ) y = -y;
            if( x >= ss_siz.x ) x = 2*(ss_siz.x-1)-x;
            if( y >= ss_siz.y ) y = 2*(ss_siz.y-1)-y;

            conv += w*p_in[ get_3d_idx(x,y,ss_idx.z,ss_siz) ];
        }

        p_out[ get_3d_idx(ss_idx,ss_siz) ] = val - conv;
    }
}

__global__ void keep_fpix_range(float*p_out, const float2 range, const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float val = p_out[ get_3d_idx(ss_idx,ss_siz) ];

        float x = ss_idx.x;
        float y = ss_idx.y-ss_siz.y/2;
        float R = l2_distance(x,y);

        float w1 = 1./(1+exp(0.5*(range.x-R)));
        float w2 = 1./(1+exp(0.5*(R-range.y)));

        p_out[ get_3d_idx(ss_idx,ss_siz) ] = val*w1*w2;
    }
}

__global__ void ctf_linearize(float*p_out, cudaTextureObject_t stk_tex, const float scale, const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float x = ss_idx.x;
        float y = ss_idx.y-ss_siz.y/2;
        float R = l2_distance(x,y);
        if( R < 0.5 ) R = 1;
        x = x/R;
        y = y/R;
        float r = R/(ss_siz.y/2);
        r = scale*ss_siz.y*sqrt(r)/2;
        x *= r;
        y *= r;

        float val = tex2DLayered<float>(stk_tex, x + 0.5, y + ss_siz.y/2 + 0.5, ss_idx.z);
        p_out[ get_3d_idx(ss_idx,ss_siz) ] = val;

    }
}

__global__ void signed_sqrt(float*p_work, const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float val = p_work[ get_3d_idx(ss_idx,ss_siz) ];
        if( val < 0 )
            val = -sqrt(-val);
        else
            val = sqrt(val);
        p_work[ get_3d_idx(ss_idx,ss_siz) ] = val;
    }
}

__global__ void tangential_blur(float*p_out, cudaTextureObject_t texture, const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float Nh = ss_siz.y/2;

        float x = ss_idx.x;
        float y = ss_idx.y-Nh;
        float R = l2_distance(x,y);
        float val = 0;
        if( R>3 ) {
            float xp,yp,ang;
            val = 0.3*tex2DLayered<float>(texture,ss_idx.x+0.5,ss_idx.y+0.5,ss_idx.z);

            ang = atan2f(1.0,R);
            xp =  cos(ang)*x + sin(ang)*y;
            yp = -sin(ang)*x + cos(ang)*y;
            if(xp<0) {xp = -xp; yp = -yp;}
            val += 0.2*tex2DLayered<float>(texture,xp+0.5,yp+Nh+0.5,ss_idx.z);

            xp =  cos(-ang)*x + sin(-ang)*y;
            yp = -sin(-ang)*x + cos(-ang)*y;
            if(xp<0) {xp = -xp; yp = -yp;}
            val += 0.2*tex2DLayered<float>(texture,xp+0.5,yp+Nh+0.5,ss_idx.z);

            ang = atan2f(2.0,R);
            xp =  cos(ang)*x + sin(ang)*y;
            yp = -sin(ang)*x + cos(ang)*y;
            if(xp<0) {xp = -xp; yp = -yp;}
            val += 0.15*tex2DLayered<float>(texture,xp+0.5,yp+Nh+0.5,ss_idx.z);

            xp =  cos(-ang)*x + sin(-ang)*y;
            yp = -sin(-ang)*x + cos(-ang)*y;
            if(xp<0) {xp = -xp; yp = -yp;}
            val += 0.15*tex2DLayered<float>(texture,xp+0.5,yp+Nh+0.5,ss_idx.z);
        }

        p_out[ get_3d_idx(ss_idx,ss_siz) ] = val;

    }
}

__global__ void radial_highpass(float*p_out, cudaTextureObject_t texture, const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float Nh = ss_siz.y/2;

        float x = ss_idx.x;
        float y = ss_idx.y-Nh;
        float R = l2_distance(x,y);
        float val = 0;

        if( R > 4 ) {
            float x_n = x/R;
            float y_n = y/R;

            for(int i=-3;i<4;i++) {
                float I = abs(i);
                I = I*I;
                float w = 0.047*I*I - 0.94*I + 2.6;
                float x_t = (float)(ss_idx.x) + i*x_n + 0.5;
                float y_t = (float)(ss_idx.y) + i*y_n + 0.5;
                val += w*tex2DLayered<float>(texture, x_t, y_t, ss_idx.z);
            }
        }

        p_out[ get_3d_idx(ss_idx,ss_siz) ] = val;

    }
}

__global__ void radial_edge_detect(float*p_out, cudaTextureObject_t texture, const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float Nh = ss_siz.y/2;

        float x = ss_idx.x;
        float y = ss_idx.y-Nh;
        float R = l2_distance(x,y);
        float val = 0;

        if( R > 2 && R < Nh-1) {
            float x_n = x/R;
            float y_n = y/R;

            float x_t = (float)(ss_idx.x) + 0.5;
            float y_t = (float)(ss_idx.y) + 0.5;

            val  = 0.2*tex2DLayered<float>(texture, x_t-x_n, y_t-y_n, ss_idx.z);
            val += 0.8*tex2DLayered<float>(texture, x_t    , y_t    , ss_idx.z);
            val -= 0.8*tex2DLayered<float>(texture, x_t+x_n, y_t+y_n, ss_idx.z);
            val -= 0.2*tex2DLayered<float>(texture, x_t+2*x_n, y_t+2*y_n, ss_idx.z);
        }

        p_out[ get_3d_idx(ss_idx,ss_siz) ] = fmaxf(val,0);

    }
}

__global__ void sum_along_z(float*p_out, const float*p_in, const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < 1 ) {

        float val = 0;
        int idx = ss_idx.x + ss_idx.y*ss_siz.x;
        int off = ss_siz.x*ss_siz.y;

        for(int k=0;k<ss_siz.z;k++) {
            val += p_in[idx];
            idx += off;
        }

        p_out[ get_3d_idx(ss_idx,ss_siz) ] = val;

    }
}

__global__ void mask_ellipsoid(float*p_work, const float4*p_def_inf, const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float Nh = ss_siz.y/2;

        float x = ss_idx.x;
        float y = ss_idx.y-Nh;
        float R = l2_distance(x,y);
        if(R<0.5) R = 1;

        x = x/R;
        y = y/R;

        float r = calc_def(x,y,p_def_inf[ss_idx.z].x,p_def_inf[ss_idx.z].y,p_def_inf[ss_idx.z].z);

        float val = 0;

        if( fabsf(r-R) < p_def_inf[ss_idx.z].w )
            val = p_work[ get_3d_idx(ss_idx,ss_siz) ];

        p_work[ get_3d_idx(ss_idx,ss_siz) ] = val;

    }
}

__global__ void rmv_bg(float*p_out,const float*p_in,const float lambda_def,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float val = 0;
        float rad = min(max(lambda_def/((float)ss_idx.x),3.0),15.0);
        int n_avg_f = (int)ceilf(rad);
        for(int i=-n_avg_f; i<=n_avg_f; i++ ) {
            int x = ss_idx.x + i;
            if( x<0 ) x = -x;
            if( x>=ss_siz.x) x = 2*(ss_siz.x-1)-x;
            val += p_in[get_3d_idx(x,ss_idx.y,ss_idx.z,ss_siz)];
        }

        int ix = get_3d_idx(ss_idx,ss_siz);
        p_out[ ix ] = p_in[ ix ] - val/(2*n_avg_f+1);
    }
}

__global__ void load_signal(float*p_out,const float*p_in,const int N,const int M,const int K) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < N && ss_idx.y < K && ss_idx.z < 1 ) {

        int idx_o = ss_idx.x + ss_idx.y*N;
        int x = ss_idx.x;
        if( x > M ) x = 2*(M-1)-x;
        int idx_i = x + ss_idx.y*M;

        float val = p_in[ idx_i ];
        p_out[ idx_o ] = val;
    }
}

__global__ void prepare_hilbert(float2*p_data,const int limit,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        int idx = get_3d_idx(ss_idx,ss_siz);
        float2 val = p_data[ idx ];

        if( ss_idx.x > 0 && ss_idx.x < limit ) {
            val.x *= 2;
            val.y *= 2;
        }
        if( ss_idx.x > limit ) {
            val.x = 0;
            val.y = 0;
        }


        p_data[ idx ] = val;
    }
}

__global__ void load_hilbert_rslt(float*p_out,const float2*p_in,const int N,const int M,const int K) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < M && ss_idx.y < K && ss_idx.z < 1 ) {

        int idx_o = ss_idx.x + ss_idx.y*N;
        int idx_i = ss_idx.x + ss_idx.y*M;

        float2 val = p_in[ idx_i ];
        float v = cuCabsf(val);
        p_out[ idx_o ] = v;
    }
}

__global__ void shift_amplitude(float*p_data,float*radial_ia,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float Nh = ss_siz.y/2;

        float x = ss_idx.x;
        float y = ss_idx.y-Nh;
        float R = l2_distance(x,y);

        int r = (int)roundf(R);
        float val = 0;
        int idx = get_3d_idx(ss_idx,ss_siz);

        if( r > 0 && r < ss_siz.x ) {
            val = p_data[ idx ];
            float wgt = radial_ia[ r + ss_idx.z*ss_siz.x ];
            val = val + (wgt/ss_siz.x);
        }

        p_data[ idx ] = val;
    }
}

__global__ void normalize_amplitude(float*p_data,float*radial_ia,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float Nh = ss_siz.y/2;

        float x = ss_idx.x;
        float y = ss_idx.y-Nh;
        float R = l2_distance(x,y);

        int r = (int)roundf(R);
        float val = 0.5;
        int idx = get_3d_idx(ss_idx,ss_siz);

        if( r > 0 && r < ss_siz.x ) {
            val = p_data[ idx ];
                        float wgt = radial_ia[ r + ss_idx.z*ss_siz.x ];
                        val = val/wgt;
                        val = val/2 + 0.5;
                        if( wgt < SUSAN_FLOAT_TOL ) val = 0.5;
        }

        p_data[ idx ] = val;
    }
}

__global__ void vis_copy_data(float*p_out,const float*p_in,const float*p_env,const float2*p_env_min,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    int N  = ss_siz.y;
    int Nh = N/2;
    int M  = Nh+1;

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float val = 0.5;
        int x = ss_idx.x-Nh;
        int y = ss_idx.y-Nh;

        float r = l2_distance(x,y);

        if( x >= 0 && r < Nh ) {
            r = fminf(r,Nh);
            float env = p_env[((int)r)+ss_idx.z*M];
            if( r > p_env_min[ss_idx.z].y ) {
                env = p_env_min[ss_idx.z].x;
            }
            if( x < 0 ) { x = -x; y = -y; }
            y = y+Nh;
            val = p_in[ x + y*M + ss_idx.z*N*M ];
            val = val/(2*env) + 0.5;
            val = fminf(fmaxf(val,0),1);
        }

        p_out[ get_3d_idx(ss_idx,ss_siz) ] = val;
    }
}

__global__ void vis_add_ctf(float*p_out,const float4*p_def_inf,const float apix,const float lambda_pi,const float lambda3_Cs_pi_2,const float ac,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    int N  = ss_siz.y;
    int Nh = N/2;

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float val = p_out[ get_3d_idx(ss_idx,ss_siz) ];
        int x = ss_idx.x-Nh;
        int y = ss_idx.y-Nh;
        float R = l2_distance(x,y);

        if( x < 0 && R < Nh ) {
            float s2 = calc_s(R,N,apix);
            s2 *= s2;
            if(R<0.5) R = 1;

            float def   = calc_def(x/R,y/R,p_def_inf[ss_idx.z].x,p_def_inf[ss_idx.z].y,p_def_inf[ss_idx.z].z);
            float gamma = calc_gamma(def,lambda_pi,lambda3_Cs_pi_2,s2,p_def_inf[ss_idx.z].w);
            float ctf   = calc_ctf(gamma,ac);
            val = ctf*ctf;
        }

        p_out[ get_3d_idx(ss_idx,ss_siz) ] = val; //min(max(val,-0.1),1.1);
    }
}

/// For reconstruction
__global__ void ctf_stk_no_correction(cudaSurfaceObject_t s_stk,cudaSurfaceObject_t s_ctf,const float2*g_data, const float3 bandpass,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 val = {0,0};
        float  ctf = 0;

        float x,y,R;
        get_xyR_unit(x,y,R,ss_idx.x,ss_idx.y-ss_siz.y/2);

        float w = get_bp_wgt(bandpass.x,bandpass.y,bandpass.z,R);

        if( w > 0.025 ) {
            val = g_data[ get_3d_idx(ss_idx,ss_siz) ];
            val.x *= w;
            val.y *= w;
            ctf = 1.0;
        }

        store_surface(s_stk,val,ss_idx);
        store_surface(s_ctf,ctf,ss_idx);

    }

}

/// For reconstruction
__global__ void ctf_stk_phase_flip( cudaSurfaceObject_t s_stk,cudaSurfaceObject_t s_ctf,const float2*g_data,
                                    const CtfConst ctf_const, const Defocus*def, const float3 bandpass,const int3 ss_siz)
{

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 val = {0,0};
        float  ctf = 0;

        float x,y,R;
        get_xyR_unit(x,y,R,ss_idx.x,ss_idx.y-ss_siz.y/2);

        float w = get_bp_wgt(bandpass.x,bandpass.y,bandpass.z,R);

        if( w > 0.025 ) {
            float s = calc_s(R,ss_siz.y,ctf_const.apix);
            float z = calc_def(x,y,def[ss_idx.z]);
            float g = calc_gamma(z,ctf_const.LambdaPi,ctf_const.CsLambda3PiH,s*s,def[ss_idx.z].ph_shft);
            ctf = calc_ctf(g,ctf_const.AC,ctf_const.CA);
            if(ctf<0)
                ctf = -1.0;
            else
                ctf =  1.0;
            val = g_data[ get_3d_idx(ss_idx,ss_siz) ];
            val.x *= w*ctf;
            val.y *= w*ctf;
            ctf = 1.0;
        }

        store_surface(s_stk,val,ss_idx);
        store_surface(s_ctf,ctf,ss_idx);

    }
}

/// For reconstruction
__global__ void ctf_stk_wiener( cudaSurfaceObject_t s_stk,cudaSurfaceObject_t s_ctf,const float2*g_data,
                                const CtfConst ctf_const,const Defocus*def,const float3 bandpass,const int3 ss_siz)
{

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 val = {0,0};
        float  ctf = 0;

        float x,y,R;
        get_xyR_unit(x,y,R,ss_idx.x,ss_idx.y-ss_siz.y/2);

        float max_R = bandpass.y;
        if( def[ss_idx.z].max_res > 0 )
            max_R = min(max_R,def[ss_idx.z].max_res);
        float w = get_bp_wgt(bandpass.x,max_R,bandpass.z,R);

        if( w > 0.025 ) {
            float s = calc_s(R,ss_siz.y,ctf_const.apix);
            float z = calc_def(x,y,def[ss_idx.z]);
            float g = calc_gamma(z,ctf_const.LambdaPi,ctf_const.CsLambda3PiH,s*s,def[ss_idx.z].ph_shft);
            ctf = calc_ctf(g,ctf_const.AC,ctf_const.CA);
            if( def[ss_idx.z].Bfactor > 0 )
                ctf *= calc_bfactor(s,def[ss_idx.z].Bfactor);
            if( def[ss_idx.z].ExpFilt > 0 )
                ctf *= calc_bfactor(s,def[ss_idx.z].ExpFilt);

            val = g_data[ get_3d_idx(ss_idx,ss_siz) ];
            val.x = w*ctf*val.x;
            val.y = w*ctf*val.y;
            ctf *= ctf;
        }

        store_surface(s_stk,val,ss_idx);
        store_surface(s_ctf,ctf,ss_idx);

    }
}

/// For reconstruction
__global__ void ctf_stk_pre_wiener( cudaSurfaceObject_t s_stk,cudaSurfaceObject_t s_ctf,const float2*g_data,
                                    const CtfConst ctf_const,const Defocus*def,const float3 bandpass,const int3 ss_siz)
{

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 val = {0,0};
        float  ctf = 0;

        float x,y,R;
        get_xyR_unit(x,y,R,ss_idx.x,ss_idx.y-ss_siz.y/2);

        float max_R = bandpass.y;
        if( def[ss_idx.z].max_res > 0 )
            max_R = min(max_R,def[ss_idx.z].max_res);
        float w = get_bp_wgt(bandpass.x,max_R,bandpass.z,R);

        if( w > 0.025 ) {
            float s = calc_s(R,ss_siz.y,ctf_const.apix);
            float z = calc_def(x,y,def[ss_idx.z]);
            float g = calc_gamma(z,ctf_const.LambdaPi,ctf_const.CsLambda3PiH,s*s,def[ss_idx.z].ph_shft);
            ctf = calc_ctf(g,ctf_const.AC,ctf_const.CA);
            if( def[ss_idx.z].Bfactor > 0 )
                ctf *= calc_bfactor(s,def[ss_idx.z].Bfactor);
            if( def[ss_idx.z].ExpFilt > 0 )
                ctf *= calc_bfactor(s,def[ss_idx.z].ExpFilt);

            val   = g_data[ get_3d_idx(ss_idx,ss_siz) ];
            val.x = w*ctf*val.x;
            val.y = w*ctf*val.y;
            ctf   = 1.0;
        }

        store_surface(s_stk,val,ss_idx);
        store_surface(s_ctf,ctf,ss_idx);

    }
}

/// For reconstruction
__global__ void ctf_stk_wiener_ssnr( cudaSurfaceObject_t s_stk,cudaSurfaceObject_t s_ctf,const float2*g_data,
                                     const CtfConst ctf_const,const Defocus*def,const float ssnr_F,const float ssnr_S,
                                     const float3 bandpass,const int3 ss_siz)
{

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 val = {0,0};
        float  ctf = 0;

        float x,y,R;
        get_xyR_unit(x,y,R,ss_idx.x,ss_idx.y-ss_siz.y/2);

        float max_R = bandpass.y;
        if( def[ss_idx.z].max_res > 0 )
            max_R = min(max_R,def[ss_idx.z].max_res);
        float w = get_bp_wgt(bandpass.x,max_R,bandpass.z,R);

        if( w > 0.025 ) {
            float s = calc_s(R,ss_siz.y,ctf_const.apix);
            float z = calc_def(x,y,def[ss_idx.z]);
            float g = calc_gamma(z,ctf_const.LambdaPi,ctf_const.CsLambda3PiH,s*s,def[ss_idx.z].ph_shft);
            ctf = calc_ctf(g,ctf_const.AC,ctf_const.CA);
            if( def[ss_idx.z].Bfactor > 0 )
                ctf *= calc_bfactor(s,def[ss_idx.z].Bfactor);
            if( def[ss_idx.z].ExpFilt > 0 )
                w *= calc_bfactor(s,def[ss_idx.z].ExpFilt);

            val = g_data[ get_3d_idx(ss_idx,ss_siz) ];
            val.x = w*ctf*val.x;
            val.y = w*ctf*val.y;
            ctf *= ctf;
            ctf += calc_ssnr(R,ssnr_F,ssnr_S);
        }

        store_surface(s_stk,val,ss_idx);
        store_surface(s_ctf,ctf,ss_idx);

    }
}

/// For CTF estimation/refinement and particle alignment
__global__ void create_ctf( float*g_ctf,const float3 delta,const CtfConst ctf_const,const Defocus*def,const int3 ss_siz) {

        int3 ss_idx = get_th_idx();

        if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

            float x,y,R;
            get_xyR_unit(x,y,R,ss_idx.x,ss_idx.y-ss_siz.y/2);

            float U = def[ss_idx.z].U + delta.x;
            float V = def[ss_idx.z].V + delta.y;
            float A = def[ss_idx.z].angle + delta.z;

            float s = calc_s(R,ss_siz.y,ctf_const.apix);
            float z = calc_def(x,y,U,V,A);
            float g = calc_gamma(z,ctf_const.LambdaPi,ctf_const.CsLambda3PiH,s*s,def[ss_idx.z].ph_shft);
            float ctf = calc_ctf(g,ctf_const.AC,ctf_const.CA);
            if( def[ss_idx.z].Bfactor > 0 )
                ctf *= calc_bfactor(s,def[ss_idx.z].Bfactor);
            g_ctf[get_3d_idx(ss_idx,ss_siz)] = ctf;

        }
}

/// For CTF estimation/refinement and particle alignment
__global__ void create_ctf( float*g_ctf,const CtfConst ctf_const,const Defocus*def,const int3 ss_siz) {

        int3 ss_idx = get_th_idx();

        if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

            float x,y,R;
            get_xyR_unit(x,y,R,ss_idx.x,ss_idx.y-ss_siz.y/2);

            float s = calc_s(R,ss_siz.y,ctf_const.apix);
            float z = calc_def(x,y,def[ss_idx.z]);
            float g = calc_gamma(z,ctf_const.LambdaPi,ctf_const.CsLambda3PiH,s*s,def[ss_idx.z].ph_shft);
            float ctf = calc_ctf(g,ctf_const.AC,ctf_const.CA);
            if( def[ss_idx.z].Bfactor > 0 )
                ctf *= calc_bfactor(s,def[ss_idx.z].Bfactor);
            g_ctf[get_3d_idx(ss_idx,ss_siz)] = ctf;

        }
}

/// Used in alignment
__global__ void mask_small_ctf( float2*g_data,const float*g_ctf,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long   ix  = get_3d_idx(ss_idx,ss_siz);
        float2 val = g_data[ ix ];
        float  ctf = g_ctf [ ix ];

        ctf = fabs(ctf);
        ctf = sqrtf(ctf);

        val.x = ctf*val.x;
        val.y = ctf*val.y;

        g_data[ix] = val;

    }
}

/// Used in alignment
__global__ void correct_stk_phase_flip( float2*g_data,const float*g_ctf,const Defocus*def,const float3 bandpass,const CtfConst ctf_const,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long ix = get_3d_idx(ss_idx,ss_siz);
        float2 val = {0,0};

        float x,y,R;
        get_xyR_unit(x,y,R,ss_idx.x,ss_idx.y-ss_siz.y/2);

        float max_R = bandpass.y;
        if( def[ss_idx.z].max_res > 0 )
            max_R = min(max_R,def[ss_idx.z].max_res);
        float w = get_bp_wgt(bandpass.x,max_R,bandpass.z,R);

        if( w > 0.05 ) {
            val = g_data[ ix ];

            float ctf = g_ctf [ ix ];
            if(ctf<0)
                ctf = -1.0;
            else
                ctf =  1.0;

            val.x = ctf*val.x;
            val.y = ctf*val.y;
        }

        g_data[ix] = val;

    }
}

/// Used in alignment
__global__ void correct_stk_wiener( float2*g_data,const float*g_ctf,const Defocus*def,const float3 bandpass,const CtfConst ctf_const,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long ix = get_3d_idx(ss_idx,ss_siz);
        float2 val = {0,0};

        float x,y,R;
        get_xyR_unit(x,y,R,ss_idx.x,ss_idx.y-ss_siz.y/2);

        float max_R = bandpass.y;
        if( def[ss_idx.z].max_res > 0 )
            max_R = min(max_R,def[ss_idx.z].max_res);
        float w = get_bp_wgt(bandpass.x,max_R,bandpass.z,R);

        if( w > 0.05 ) {
            val = g_data[ ix ];

            float ctf = g_ctf [ ix ];

            float s = calc_s(R,ss_siz.y,ctf_const.apix);;
            if( def[ss_idx.z].ExpFilt > 0 )
                w *= calc_bfactor(s,def[ss_idx.z].ExpFilt);

            val.x = w*ctf*val.x;
            val.y = w*ctf*val.y;
            ctf  *= ctf;
            val.x = val.x/(ctf+0.001);
            val.y = val.y/(ctf+0.001);
        }

        g_data[ix] = val;

    }
}

/// Used in alignment
__global__ void correct_stk_wiener_ssnr( float2*g_data,const float*g_ctf,const Defocus*def,const float ssnr_F,const float ssnr_S,const float3 bandpass,const CtfConst ctf_const,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long ix = get_3d_idx(ss_idx,ss_siz);
        float2 val = {0,0};

        float x,y,R;
        get_xyR_unit(x,y,R,ss_idx.x,ss_idx.y-ss_siz.y/2);

        float max_R = bandpass.y;
        if( def[ss_idx.z].max_res > 0 )
        max_R = min(max_R,def[ss_idx.z].max_res);
        float w = get_bp_wgt(bandpass.x,max_R,bandpass.z,R);

        if( w > 0.05 ) {
            val = g_data[ ix ];

            float ctf = g_ctf [ ix ];

            float s = calc_s(R,ss_siz.y,ctf_const.apix);;
            if( def[ss_idx.z].ExpFilt > 0 )
                w *= calc_bfactor(s,def[ss_idx.z].ExpFilt);

            val.x = w*ctf*val.x;
            val.y = w*ctf*val.y;
            ctf  *= ctf;
            ctf  += calc_ssnr(R,ssnr_F,ssnr_S);
            val.x = val.x/(ctf+0.001);
            val.y = val.y/(ctf+0.001);
        }

        g_data[ix] = val;

    }
}

}

#endif /// GPU_KERNEL_CTF_H


