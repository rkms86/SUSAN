/*
 * This file is part of the Substack Analysis (SUSAN) framework.
 * Copyright (c) 2018-2021 Ricardo Miguel Sanchez Loayza.
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

#ifndef CTF_NORMALIZER_H
#define CTF_NORMALIZER_H

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

class CtfNormalizer {
public:
	int3  ss_siz;
	dim3  blk;
	dim3  grd;
	GPU::Stream       stream;
	GPU::GArrDouble   ss_acc;
	GPU::GArrDouble   ss_wgt;
	GPU::GArrSingle   ss_norm;
	GPU::GArrSingle3  ss_vec_r;
	GPU::GArrSingle   ss_ctf_ps;
	GPU::GArrSingle   ss_acc_avg;
	GPU::GArrSingle   ss_acc_std;
	GPU::GArrSingle2  ss_fourier;
	GpuFFT::FFT2D     fft2;
	GPU::GTex2DSingle ss_ps;
		
	CtfNormalizer(int N,int K) {
		stream.configure();
		
		ss_siz = make_int3((N/2)+1,N,K);
		blk = GPU::get_block_size_2D();
		grd = GPU::calc_grid_size(blk,ss_siz.x,ss_siz.y,ss_siz.z);
		
		ss_ps.alloc(ss_siz.x,ss_siz.y,ss_siz.z);
		ss_acc.alloc(ss_siz.x*ss_siz.y*ss_siz.z);
		ss_wgt.alloc(ss_siz.x*ss_siz.y*ss_siz.z);
		ss_norm.alloc(ss_siz.x*ss_siz.y*ss_siz.z);
		ss_vec_r.alloc(ss_siz.x*ss_siz.y);
		ss_ctf_ps.alloc(ss_siz.x*ss_siz.y*ss_siz.z);
		ss_acc_avg.alloc(ss_siz.z);
		ss_acc_std.alloc(ss_siz.z);
		ss_fourier.alloc(ss_siz.x*ss_siz.y*ss_siz.z);
		
		dim3 grd2D = GPU::calc_grid_size(blk,ss_siz.x,ss_siz.y,1);
		GpuKernelsCtf::create_vec_r<<<grd2D,blk,0,stream.strm>>>(ss_vec_r.ptr,ss_siz);
		
		fft2.alloc(ss_siz.x,ss_siz.y,ss_siz.z);
		fft2.set_stream(stream.strm);
	}
	
	void clear_acc() {
		ss_acc.clear(stream.strm);
		ss_wgt.clear(stream.strm);
		stream.sync();
	}
	
	void process(float*p_data,float2*p_pi_lambda_dZ,float apix,float bin_factor) {
		fft2.exec(ss_fourier.ptr,p_data);
		fftshift();
		load_ps();
		normalize(p_pi_lambda_dZ,apix,bin_factor);
		accumulate();
		stream.sync();
	}
	
	void apply_wgt() {
		GpuKernelsCtf::divide<<<grd,blk,0,stream.strm>>>(ss_ctf_ps.ptr,ss_acc.ptr,ss_wgt.ptr,ss_siz);
	}
	
	void download(single*c_rslt) {
		GPU::download_async(c_rslt,ss_ctf_ps.ptr,ss_siz.x*ss_siz.y*ss_siz.z,stream.strm);
	}
	
protected:
	void fftshift() {
		GpuKernels::fftshift2D<<<grd,blk,0,stream.strm>>>(ss_fourier.ptr,ss_siz);
	}
	
	void load_ps() {
		GpuKernels::load_surf_abs<<<grd,blk,0,stream.strm>>>(ss_ps.surface,ss_fourier.ptr,ss_siz);
	}
	
	void normalize(float2*p_pi_lambda_dZ,float apix,float bin_factor) {
		ss_acc_avg.clear(stream.strm);
		ss_acc_std.clear(stream.strm);
		GpuKernelsCtf::ctf_normalize<<<grd,blk,0,stream.strm>>>(ss_ctf_ps.ptr,ss_ps.texture,p_pi_lambda_dZ,ss_vec_r.ptr,apix,bin_factor,ss_siz);
		GpuKernels::get_avg_std<<<grd,blk,0,stream.strm>>>(ss_acc_std.ptr,ss_acc_avg.ptr,ss_ctf_ps.ptr,ss_siz);
		GpuKernels::zero_avg_one_std<<<grd,blk,0,stream.strm>>>(ss_ctf_ps.ptr,ss_acc_std.ptr,ss_acc_avg.ptr,ss_siz);
                //GpuKernels::conv_gaussian<<<grd,blk,0,stream.strm>>>(ss_norm.ptr,ss_ctf_ps.ptr,ss_siz);
                //GpuKernels::stk_medfilt<<<grd,blk,0,stream.strm>>>(ss_norm.ptr,ss_ctf_ps.ptr,ss_siz);
	}
	
	void accumulate() {
                GpuKernelsCtf::accumulate<<<grd,blk,0,stream.strm>>>(ss_acc.ptr,ss_wgt.ptr,ss_ctf_ps.ptr,ss_siz);
	}
	
};

#endif /// CTF_NORMALIZER_H


