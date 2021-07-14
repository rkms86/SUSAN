#ifndef CTF_REFINER_H
#define CTF_REFINER_H

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

class CtfRefinerRadial {
public:
    int N;
    int M;
    int maxK;

    GPU::GArrSingle2 ss_fourier;
    GPU::GArrSingle  ss_norm_ast;
    GPU::GArrSingle  rad_avg;
    GPU::GArrSingle  rad_wgt;

    GPU::GTex2DSingle ss_abs_f;

    GpuFFT::FFT2D fft2;

    CtfRefinerRadial(int m,int n,int k,GPU::Stream&stream) {
        N  = n;
        M  = m;
        maxK = k;

        ss_fourier.alloc(M*N*maxK);
        ss_norm_ast.alloc(M*N*maxK);

        rad_avg.alloc(M*maxK);
        rad_wgt.alloc(M*maxK);

        ss_abs_f.alloc(M,N,maxK);
        fft2.alloc(M,N,maxK);
        fft2.set_stream(stream.strm);
    }

    ~CtfRefinerRadial() {

    }

    void add_data(GPU::GArrSingle&g_data,int k,GPU::Stream&stream) {
        int3 ss_raw = make_int3(N,N,k);
        int3 ss_fou = make_int3(M,N,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd_raw = GPU::calc_grid_size(blk,N,N,k);
        dim3 grd_fou = GPU::calc_grid_size(blk,M,N,k);
        GpuKernels::fftshift2D<<<grd_raw,blk,0,stream.strm>>>(g_data.ptr,ss_raw);
        fft2.exec(ss_fourier.ptr,g_data.ptr);
        GpuKernels::fftshift2D<<<grd_fou,blk,0,stream.strm>>>(ss_fourier.ptr,ss_fou);
        GpuKernels::load_surf_abs<<<grd_fou,blk,0,stream.strm>>>(ss_abs_f.surface,ss_fourier.ptr,ss_fou);
    }

    void compensate_astigmatism(float pi_lambda_deltaZ,float delta_ang, float apix,int k,GPU::Stream&stream) {
        int3 ss_fou = make_int3(M,N,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd_fou = GPU::calc_grid_size(blk,M,N,k);
        GpuKernelsCtf::ctf_radial_normalize<<<grd_fou,blk,0,stream.strm>>>(ss_norm_ast.ptr,ss_abs_f.texture,pi_lambda_deltaZ,delta_ang,apix,ss_fou);
    }

    void calc_radial_average(int k,GPU::Stream&stream) {
        rad_avg.clear(stream.strm);
        rad_wgt.clear(stream.strm);
        int3 ss_fou = make_int3(M,N,k);
        int3 ss_2   = make_int3(M,k,1);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd_fou = GPU::calc_grid_size(blk,M,N,k);
        GpuKernels::radial_ps_avg<<<grd_fou,blk,0,stream.strm>>>(rad_avg.ptr,rad_wgt.ptr,ss_norm_ast.ptr,ss_fou);
        GpuKernels::divide<<<grd_fou,blk,0,stream.strm>>>(rad_avg.ptr,rad_wgt.ptr,ss_2);
    }

protected:

};

#endif /// CTF_REFINER_H


