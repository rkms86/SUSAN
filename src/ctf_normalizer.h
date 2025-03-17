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
#include "estimate_ctf_args.h"
#include <iostream>

class CtfNormalizer {
public:
    int3  ss_siz;
    int3  ss_buf;
    dim3  blk;
    dim3  grd;
    dim3  grd_buf;
    GPU::Stream       stream;
    GPU::GArrDouble   ss_acc;
    GPU::GArrDouble   ss_wgt;
    GPU::GArrSingle   ss_norm;
    GPU::GArrSingle3  ss_vec_r;
    GPU::GArrSingle   ss_buffer;
    GPU::GArrSingle   ss_ctf_ps;
    GPU::GArrSingle   ss_acc_avg;
    GPU::GArrSingle   ss_acc_std;
    GPU::GArrSingle2  ss_fourier;
    GpuFFT::FFT2D     fft2;
    GPU::GTex2DSingle ss_ps;

    CtfNormalizer(int N,int K) {
        stream.configure();

        ss_siz  = make_int3((N/2)+1,N,K);
        ss_buf  = make_int3(N,N,K);
        blk     = GPU::get_block_size_2D();
        grd     = GPU::calc_grid_size(blk,ss_siz.x,ss_siz.y,ss_siz.z);
        grd_buf = GPU::calc_grid_size(blk,ss_buf.x,ss_buf.y,ss_buf.z);

        ss_ps.alloc(ss_siz.x,ss_siz.y,ss_siz.z);
        ss_acc.alloc(ss_siz.x*ss_siz.y*ss_siz.z);
        ss_wgt.alloc(ss_siz.x*ss_siz.y*ss_siz.z);
        ss_norm.alloc(ss_siz.x*ss_siz.y*ss_siz.z);
        ss_vec_r.alloc(ss_siz.x*ss_siz.y);
        ss_buffer.alloc(ss_buf.x*ss_buf.y*ss_buf.z);
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

    void average(float*p_data,float bin_factor) {
        fft2.exec(ss_fourier.ptr,p_data);
        fftshift();
        load_ps();
        bin(bin_factor);
        accumulate();
        stream.sync();

        rot180(p_data);
        fft2.exec(ss_fourier.ptr,ss_buffer.ptr);
        fftshift();
        load_ps();
        bin(bin_factor);
        accumulate();

        stream.sync();
    }

    void normal(float*p_data,float2*p_factor,float bin_factor) {
        fft2.exec(ss_fourier.ptr,p_data);
        fftshift();
        load_ps();
        normalize(p_factor,bin_factor);
        accumulate();
        stream.sync();

        rot180(p_data);
        fft2.exec(ss_fourier.ptr,ss_buffer.ptr);
        fftshift();
        load_ps();
        normalize(p_factor,bin_factor);
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
    void rot180(float*p_data) {
        GpuKernels::rotate_180_stk<<<grd_buf,blk,0,stream.strm>>>(ss_buffer.ptr,p_data,ss_buf);
    }

    void fftshift() {
        GpuKernels::fftshift2D<<<grd,blk,0,stream.strm>>>(ss_fourier.ptr,ss_siz);
    }

    void load_ps() {
        GpuKernels::load_surf_abs<<<grd,blk,0,stream.strm>>>(ss_ps.surface,ss_fourier.ptr,ss_siz);
    }

    void bin(float bin_factor) {
        ss_acc_avg.clear(stream.strm);
        ss_acc_std.clear(stream.strm);
        GpuKernelsCtf::ctf_bin<<<grd,blk,0,stream.strm>>>(ss_ctf_ps.ptr,ss_ps.texture,bin_factor,ss_siz);
        GpuKernels::get_avg_std<<<grd,blk,0,stream.strm>>>(ss_acc_std.ptr,ss_acc_avg.ptr,ss_ctf_ps.ptr,ss_siz);
        GpuKernels::zero_avg_one_std<<<grd,blk,0,stream.strm>>>(ss_ctf_ps.ptr,ss_acc_std.ptr,ss_acc_avg.ptr,ss_siz);
    }

    void normalize(float2*p_factor,float bin_factor) {
        ss_acc_avg.clear(stream.strm);
        ss_acc_std.clear(stream.strm);
        GpuKernelsCtf::ctf_normalize<<<grd,blk,0,stream.strm>>>(ss_ctf_ps.ptr,ss_ps.texture,p_factor,ss_vec_r.ptr,bin_factor,ss_siz);
        GpuKernels::get_avg_std<<<grd,blk,0,stream.strm>>>(ss_acc_std.ptr,ss_acc_avg.ptr,ss_ctf_ps.ptr,ss_siz);
        GpuKernels::zero_avg_one_std<<<grd,blk,0,stream.strm>>>(ss_ctf_ps.ptr,ss_acc_std.ptr,ss_acc_avg.ptr,ss_siz);
        GpuKernels::load_surf<<<grd,blk,0,stream.strm>>>(ss_ps.surface,ss_ctf_ps.ptr,ss_siz);
        GpuKernelsCtf::tangential_blur<<<grd,blk,0,stream.strm>>>(ss_norm.ptr,ss_ps.texture,ss_siz);
        GpuKernels::conv_gaussian<<<grd,blk,0,stream.strm>>>(ss_ctf_ps.ptr,ss_norm.ptr,0.5000,6.2831,ss_siz);
    }

    void accumulate() {
        GpuKernelsCtf::accumulate<<<grd,blk,0,stream.strm>>>(ss_acc.ptr,ss_wgt.ptr,ss_ctf_ps.ptr,ss_siz);
    }

};

#endif /// CTF_NORMALIZER_H


