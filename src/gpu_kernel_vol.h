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

__device__ void  rot_pt(float&x,float&y,float&z,const Rot33&R,const Vec3&in) {
    x = R.xx*in.x + R.xy*in.y + R.xz*in.z;
    y = R.yx*in.x + R.yy*in.y + R.yz*in.z;
    z = R.zx*in.x + R.zy*in.y + R.zz*in.z;
}

__device__ void  rot_pt_XY(Vec3&out,const Rot33&R,const Vec3&in) {
    out.x = R.xx*in.x + R.xy*in.y;
    out.y = R.yx*in.x + R.yy*in.y;
    out.z = R.zx*in.x + R.zy*in.y;
}

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

__device__ int get_ix_3d(bool&should_read,bool&should_conj,const int x,const int y,const int z,const int M,const int N) {
	int wx=x;
	int wy=y;
	int wz=z;
	should_conj = false;
        should_read = false;
	
	if(x<0) {
		wx = -wx;
		wy = -wy;
		wz = -wz;
		should_conj = true;
	}
	
	wy += N/2;
	wz += N/2;

        if( wx < M && wy < N && wz < N )
            should_read = true;
	
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

        if( bp > 0.025 ) {

            bool should_add = false;
            double  wgt = 0;
            double2 val = {0,0};
            float x,y,z;

            for(int k=0; k<K; k++ ) {
                if( pTlt[k].w > 0 ){
                    z = rot_inv_pt_Z(pTlt[k].R,pt);
                    if( z >= 0 && z<= 1 ) {
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
                        val.x += pTlt[k].w*(1-z)*read_stk.x;
                        val.y += pTlt[k].w*(1-z)*read_stk.y;
                        float  read_wgt = tex2DLayered<float >(ss_wgt, x+0.5, y+N/2+0.5, k);
                        wgt   += pTlt[k].w*(1-z)*read_wgt;

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

__global__ void insert_stk_atomic(double2*p_acc,double*p_wgt,
                                  cudaTextureObject_t ss_stk, cudaTextureObject_t ss_wgt, const Proj2D*pTlt,
                                  const float3 bandpass,const int M, const int N, const int K)
{
	int3 ss_idx = get_th_idx();
	
	if( ss_idx.x < M && ss_idx.y < N && ss_idx.z < K ) {
		
		if( pTlt[ss_idx.z].w > 0 ) {
			Vec3 pt;
			pt.x = ss_idx.x;
			pt.y = ss_idx.y - N/2;
			pt.z = 0;

			float R = sqrt( pt.x*pt.x + pt.y*pt.y );
			float bp = get_bp_wgt(bandpass.x,bandpass.y,bandpass.z,R);

			if( bp > 0.025 ) {
				
				float2 val = tex2DLayered<float2>(ss_stk, float(ss_idx.x)+0.5, float(ss_idx.y)+0.5, ss_idx.z);
				float  wgt = tex2DLayered<float >(ss_wgt, float(ss_idx.x)+0.5, float(ss_idx.y)+0.5, ss_idx.z);
				float x,y,z;
                val.x *= bp*pTlt[ss_idx.z].w;
                val.y *= bp*pTlt[ss_idx.z].w;
                wgt   *= bp*pTlt[ss_idx.z].w;
				
				rot_pt(x,y,z,pTlt[ss_idx.z].R,pt);
				if( x < 0 ) {
					x = -x;
					y = -y;
					z = -z;
					val.y = -val.y;
				}
				
				y += N/2;
				z += N/2;

				int x0 = int( floor(x) );
				int y0 = int( floor(y) );
				int z0 = int( floor(z) );
				int x1 = x0 + 1;
				int y1 = y0 + 1;
				int z1 = z0 + 1;
				
				float wx1 = x - floor(x);
				float wy1 = y - floor(y);
				float wz1 = z - floor(z);
				float wx0 = 1 - wx1;
				float wy0 = 1 - wy1;
				float wz0 = 1 - wz1;
				
				bool bx0 = x0 < M;
				bool bx1 = x1 < M;
				bool by0 = (y0>=0) && (y0<N);
				bool by1 = (y1>=0) && (y1<N);
				bool bz0 = (z0>=0) && (z0<N);
				bool bz1 = (z1>=0) && (z1<N);
				
				long idx;
				double w_wgt;
				
				if( bx0 && by0 && bz0 ) {
					idx = x0 + y0*M + z0*M*N;
					w_wgt = wx0*wy0*wz0;
					atomic_Add( &(p_acc[idx].x) , w_wgt*val.x );
					atomic_Add( &(p_acc[idx].y) , w_wgt*val.y );
					atomic_Add( &(p_wgt[idx]  ) , w_wgt*wgt   );
				}
				
				if( bx1 && by0 && bz0 ) {
					idx = x1 + y0*M + z0*M*N;
					w_wgt = wx1*wy0*wz0;
					atomic_Add( &(p_acc[idx].x) , w_wgt*val.x );
					atomic_Add( &(p_acc[idx].y) , w_wgt*val.y );
					atomic_Add( &(p_wgt[idx]  ) , w_wgt*wgt   );
				}
				
				if( bx0 && by1 && bz0 ) {
					idx = x0 + y1*M + z0*M*N;
					w_wgt = wx0*wy1*wz0;
					atomic_Add( &(p_acc[idx].x) , w_wgt*val.x );
					atomic_Add( &(p_acc[idx].y) , w_wgt*val.y );
					atomic_Add( &(p_wgt[idx]  ) , w_wgt*wgt   );
				}
				
				if( bx1 && by1 && bz0 ) {
					idx = x1 + y1*M + z0*M*N;
					w_wgt = wx1*wy1*wz0;
					atomic_Add( &(p_acc[idx].x) , w_wgt*val.x );
					atomic_Add( &(p_acc[idx].y) , w_wgt*val.y );
					atomic_Add( &(p_wgt[idx]  ) , w_wgt*wgt   );
				}
				
				if( bx0 && by0 && bz1 ) {
					idx = x0 + y0*M + z1*M*N;
					w_wgt = wx0*wy0*wz1;
					atomic_Add( &(p_acc[idx].x) , w_wgt*val.x );
					atomic_Add( &(p_acc[idx].y) , w_wgt*val.y );
					atomic_Add( &(p_wgt[idx]  ) , w_wgt*wgt   );
				}
				
				if( bx1 && by0 && bz1 ) {
					idx = x1 + y0*M + z1*M*N;
					w_wgt = wx1*wy0*wz1;
					atomic_Add( &(p_acc[idx].x) , w_wgt*val.x );
					atomic_Add( &(p_acc[idx].y) , w_wgt*val.y );
					atomic_Add( &(p_wgt[idx]  ) , w_wgt*wgt   );
				}
				
				if( bx0 && by1 && bz1 ) {
					idx = x0 + y1*M + z1*M*N;
					w_wgt = wx0*wy1*wz1;
					atomic_Add( &(p_acc[idx].x) , w_wgt*val.x );
					atomic_Add( &(p_acc[idx].y) , w_wgt*val.y );
					atomic_Add( &(p_wgt[idx]  ) , w_wgt*wgt   );
				}
				
				if( bx1 && by1 && bz1 ) {
					idx = x1 + y1*M + z1*M*N;
					w_wgt = wx1*wy1*wz1;
					atomic_Add( &(p_acc[idx].x) , w_wgt*val.x );
					atomic_Add( &(p_acc[idx].y) , w_wgt*val.y );
					atomic_Add( &(p_wgt[idx]  ) , w_wgt*wgt   );
				}
			
			}
		}
	}
}

__global__ void extract_stk(float2*p_out,cudaTextureObject_t vol,const Proj2D*pTlt,
                            const float3 bandpass,const int M, const int N, const int K)
{
    int3 ss_idx = get_th_idx();

    if( ss_idx.x < M && ss_idx.y < N && ss_idx.z < K ) {

        float2 val = {0,0};

        if( pTlt[ss_idx.z].w > 0 ) {
            Vec3 pt_in;
            pt_in.x = ss_idx.x;
            pt_in.y = ss_idx.y - N/2;
            pt_in.z = 0;

            float R = l2_distance(pt_in.x,pt_in.y);
            float bp = get_bp_wgt(bandpass.x,bandpass.y,bandpass.z,R);

            if( bp > 0.05 ) {
                Vec3 pt_out;
                rot_pt_XY(pt_out,pTlt[ss_idx.z].R,pt_in);

                bool should_conjugate = false;
                if( pt_out.x < 0 ) {
                    pt_out.x = -pt_out.x;
                    pt_out.y = -pt_out.y;
                    pt_out.z = -pt_out.z;
                    should_conjugate = true;
                }

                val = tex3D<float2>(vol, pt_out.x+0.5, pt_out.y+N/2+0.5, pt_out.z+N/2+0.5);

                if( should_conjugate )
                    val.y = -val.y;

                val.x *= bp;
                val.y *= bp;

            }
        }

        p_out[ ss_idx.x + M*ss_idx.y + M*N*ss_idx.z ] = val;
    }
}

__global__ void invert_wgt(double*p_data,const int3 ss_siz) {
	
    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long idx = get_3d_idx(ss_idx,ss_siz);

        double data = p_data[idx];

        if( abs(data) < 0.025 ) {
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

__global__ void inv_wgt_ite_hard_shrink(double*p_vol_wgt,double min_wgt,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {
        long idx = get_3d_idx(ss_idx,ss_siz);
        double w = p_vol_wgt[idx];
        if( w < min_wgt && w > 0)
            w = min_wgt;
        p_vol_wgt[ idx ] = w;
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
        den = copysignf(fmax(abs(den),1e-2),den);
        p_vol_wgt[ idx ] = p_vol_wgt[ idx ] / den;
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
        float sinc_coef = ( arg > 0.001 ) ? sinf(arg)/arg : 1.0;
        sinc_coef *= sinc_coef;

        float val = p_data[ix];
        p_data[ix] = val/sinc_coef;
    }
}

__global__ void boost_low_freq(float2*p_out,
                               const float scale, const float value, const float decay,
                               const int3 ss_siz)
{
	int3 ss_idx = get_th_idx();

	if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

		long ix = ss_idx.x + ss_idx.y*ss_siz.x + ss_idx.z*ss_siz.x*ss_siz.y;
		int center = ss_siz.y/2;
		float R = l2_distance(ss_idx.x,ss_idx.y-center,ss_idx.z-center);
		
		float2 val = p_out[ix];
		
		float bp = get_bp_wgt(0,value,decay,R);
		bp = ((scale*bp)+1)/(scale+1);
		val.x *= bp;
		val.y *= bp;
		
		p_out[ ix ] = val;
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
            bool should_read = false;

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

            read_idx = get_ix_3d(should_read,should_conj,p_x  ,p_y  ,p_z  ,M,N);
            read_val = t_val[read_idx];
            read_wgt = t_wgt[read_idx];
            if(should_conj) read_val.y = -read_val.y;
            w = (1-w_x)*(1-w_y)*(1-w_z);
            val.x += w*read_val.x;
            val.y += w*read_val.y;
            wgt   += w*read_wgt;

            read_idx = get_ix_3d(should_read,should_conj,p_x+1,p_y  ,p_z  ,M,N);
            if( should_read ) {
                read_val = t_val[read_idx];
                read_wgt = t_wgt[read_idx];
                if(should_conj) read_val.y = -read_val.y;
                w = (  w_x)*(1-w_y)*(1-w_z);
                val.x += w*read_val.x;
                val.y += w*read_val.y;
                wgt   += w*read_wgt;
            }

            read_idx = get_ix_3d(should_read,should_conj,p_x  ,p_y+1,p_z  ,M,N);
            if( should_read ) {
                read_val = t_val[read_idx];
                read_wgt = t_wgt[read_idx];
                if(should_conj) read_val.y = -read_val.y;
                w = (1-w_x)*(  w_y)*(1-w_z);
                val.x += w*read_val.x;
                val.y += w*read_val.y;
                wgt   += w*read_wgt;
            }

            read_idx = get_ix_3d(should_read,should_conj,p_x+1,p_y+1,p_z  ,M,N);
            if( should_read ) {
                read_val = t_val[read_idx];
                read_wgt = t_wgt[read_idx];
                if(should_conj) read_val.y = -read_val.y;
                w = (  w_x)*(  w_y)*(1-w_z);
                val.x += w*read_val.x;
                val.y += w*read_val.y;
                wgt   += w*read_wgt;
            }

            read_idx = get_ix_3d(should_read,should_conj,p_x  ,p_y  ,p_z+1,M,N);
            if( should_read ) {
                read_val = t_val[read_idx];
                read_wgt = t_wgt[read_idx];
                if(should_conj) read_val.y = -read_val.y;
                w = (1-w_x)*(1-w_y)*(  w_z);
                val.x += w*read_val.x;
                val.y += w*read_val.y;
                wgt   += w*read_wgt;
            }

            read_idx = get_ix_3d(should_read,should_conj,p_x+1,p_y  ,p_z+1,M,N);
            if( should_read ) {
                read_val = t_val[read_idx];
                read_wgt = t_wgt[read_idx];
                if(should_conj) read_val.y = -read_val.y;
                w = (  w_x)*(1-w_y)*(  w_z);
                val.x += w*read_val.x;
                val.y += w*read_val.y;
                wgt   += w*read_wgt;
            }

            read_idx = get_ix_3d(should_read,should_conj,p_x  ,p_y+1,p_z+1,M,N);
            if( should_read ) {
                read_val = t_val[read_idx];
                read_wgt = t_wgt[read_idx];
                if(should_conj) read_val.y = -read_val.y;
                w = (1-w_x)*(  w_y)*(  w_z);
                val.x += w*read_val.x;
                val.y += w*read_val.y;
                wgt   += w*read_wgt;
            }

            read_idx = get_ix_3d(should_read,should_conj,p_x+1,p_y+1,p_z+1,M,N);
            if( should_read ) {
                read_val = t_val[read_idx];
                read_wgt = t_wgt[read_idx];
                if(should_conj) read_val.y = -read_val.y;
                w = (  w_x)*(  w_y)*(  w_z);
                val.x += w*read_val.x;
                val.y += w*read_val.y;
                wgt   += w*read_wgt;
            }

            p_val[idx] = val;
            p_wgt[idx] = wgt;
        }
    }

}

__global__ void reconstruct_pts(float*p_cc,const Proj2D*pTlt,cudaTextureObject_t ss_cc,
                                const Vec3*p_pts,const int n_pts,const int N,const int K) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < n_pts && ss_idx.y < 1 && ss_idx.z < 1 ) {

        float cc = 0;
        Vec3  pt = p_pts[ss_idx.x];
        single x,y;
        single off = (single)(N/2) + 0.5;
        
        for(int z=0;z<K;z++) {
            if( pTlt[z].w > 0 ) {
                rot_inv_pt_XY(x,y,pTlt[z].R,pt);
                cc += pTlt[z].w*tex2DLayered<float>(ss_cc,x+off,y+off,z);
            }
        }
        
        p_cc[ss_idx.x] = cc;

    }


}


__global__ void extract_pts(float*p_cc,const float*p_data,const Vec3*p_pts,const int n_pts,const int N,const int K) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < n_pts && ss_idx.y < 1 && ss_idx.z < K ) {

        Vec3  pt = p_pts[ss_idx.x];

        int x = (int)pt.x + N/2;
        int y = (int)pt.y + N/2;

        float cc = p_data[x + y*N + ss_idx.z*N*N];

        p_cc[ss_idx.x + n_pts*ss_idx.z] = cc;

    }


}

__global__ void multiply_vol2(float2*p_out,cudaTextureObject_t vol_tex,const float2*p_data,
                              const Rot33 Rot,const float3 bandpass,const int M, const int N,const float den=1)
{
    int3 ss_idx = get_th_idx();

    if( ss_idx.x < M && ss_idx.y < N && ss_idx.z < N ) {

        long idx = ss_idx.x + ss_idx.y*M + ss_idx.z*M*N;

        float2 val = {0,0};

        Vec3 pt_in;
        pt_in.x = ss_idx.x;
        pt_in.y = ss_idx.y - N/2;
        pt_in.z = ss_idx.z - N/2;

        float R = l2_distance(pt_in.x,pt_in.y,pt_in.z);
        float w = get_bp_wgt(bandpass.x,bandpass.y,bandpass.z,R);

        if( w > 0.05 ) {
            w = w/den;
            float x,y,z;
            rot_pt(x,y,z,Rot,pt_in);

            bool should_conj=false;
            if( x < 0 ) {
                x = -x;
                y = -y;
                z = -z;
                should_conj=true;
            }

            float2 val_a = tex3D<float2>(vol_tex,x+0.5,y+N/2+0.5,z+N/2+0.5);

            if( should_conj )
                val_a.y = -val_a.y;

            float2 val_b = p_data[idx];

            val = cuCmulf(val_a,val_b);
            val.x *= w;
            val.y *= w;
        }

        p_out[ idx ] = val;
    }
}

__global__ void extract_pts(float*p_cc,const float*p_data,const Vec3*p_pts,const int n_pts,const int N) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < n_pts && ss_idx.y < 1 && ss_idx.z < 1 ) {

        Vec3  pt = p_pts[ss_idx.x];

        int x = (int)pt.x + N/2;
        int y = (int)pt.y + N/2;
        int z = (int)pt.z + N/2;

        float cc = p_data[x + y*N + z*N*N];

        p_cc[ss_idx.x] = cc;

    }


}

__global__ void radial_cc(float*p_acc,const float2*p_vol_a,const float2*p_vol_b,const int M,const int N,const float scale=1) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < M && ss_idx.y < N && ss_idx.z < N ) {

        int center = N/2;
        float R = l2_distance(ss_idx.x,ss_idx.y-center,ss_idx.z-center);
        int r = round(R);

        if( r<M ) {
            long ix = ss_idx.x + ss_idx.y*M + ss_idx.z*M*N;
            float2 v_a = p_vol_a[ix];
            float2 v_b = p_vol_b[ix];
            v_b.y = -v_b.y;
            float2 val = cuCmulf(v_a,v_b);
            val.x = val.x*scale;
            atomicAdd(p_acc+r,val.x);
        }
    }
}

__global__ void calc_fsc(float*p_fsc,const float*p_den_a,const float*p_den_b,const int M) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < M && ss_idx.y < 1 && ss_idx.z < 1 ) {

        float num = p_fsc[ss_idx.x];
        float den = p_den_a[ss_idx.x]*p_den_b[ss_idx.x];
        if(den<0)
            den =1;
        den = sqrt(den);
        if(den<1e-9 && abs(num)<1e-9) {
            num = 1;
            den = 1;
        }

        p_fsc[ss_idx.x] = num/den;
    }
}

__global__ void randomize_phase(float2*p_vol,const float*p_ang,const float fpix,const int M,const int N) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < M && ss_idx.y < N && ss_idx.z < N ) {

        long ix = ss_idx.x + ss_idx.y*M + ss_idx.z*M*N;
        float2 vol = p_vol[ix];

        int center = N/2;
        float R = l2_distance(ss_idx.x,ss_idx.y-center,ss_idx.z-center);
        int r = round(R);

        if( r>=fpix ) {
            float ang = p_ang[ix];
            float2 v_a = vol;
            float2 v_b;
            v_b.x = cos(ang);
            v_b.y = sin(ang);
            vol = cuCmulf(v_b,v_a);
        }
        p_vol[ix]=vol;
    }
}


}

#endif /// GPU_KERNEL_VOL_H



