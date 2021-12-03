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

#ifndef REFS_ALIGNER_H
#define REFS_ALIGNER_H

#include <iostream>
#include "datatypes.h"
#include "reference.h"
#include "gpu.h"
#include "gpu_fft.h"
#include "gpu_kernel.h"
#include "gpu_kernel_vol.h"
#include "mrc.h"
#include "io.h"
#include "svg.h"
#include "refs_fsc_args.h"

class RefsFsc {

public:
    char*out_dir;

    int gpu_ix;
    int N;
    int M;
    float apix;
    float rand_fpix;
    float threshold;

    float *c_vol_a;
    float *c_vol_b;
    float *c_mask;

    float *fsc_unmasked;
    float *fsc_masked;
    float *fsc_randomized;

    bool save_svg;

    GPU::GArrSingle  rand_ang_a;
    GPU::GArrSingle  rand_ang_b;
    GPU::GArrSingle  real_buffer;
    GPU::GArrSingle2 vol_a;
    GPU::GArrSingle2 vol_b;

    GPU::GArrSingle  rad_num;
    GPU::GArrSingle  rad_den_a;
    GPU::GArrSingle  rad_den_b;

    GpuFFT::FFT3D  fft3;

    RefsFsc(ArgsRefsFsc::Info&info) {
        save_svg = info.save_svg;
        out_dir = info.out_dir;
        threshold = info.threshold;
        gpu_ix = info.gpu_ix;
        N = info.box_size;
        M = (N/2)+1;
        apix = info.pix_size;
        rand_fpix = info.rand_fpix;
        c_vol_a = new float[N*N*N];
        c_vol_b = new float[N*N*N];
        c_mask  = new float[N*N*N];
        fsc_unmasked   = new float[M];
        fsc_masked     = new float[M];
        fsc_randomized = new float[M];
        setup_gpu();
    }

    ~RefsFsc() {
        delete [] c_vol_a;
        delete [] c_vol_b;
        delete [] c_mask;
        delete [] fsc_unmasked;
        delete [] fsc_masked;
        delete [] fsc_randomized;
    }

    void calc_fsc(References&refs) {
        char report[SUSAN_FILENAME_LENGTH];
        sprintf(report,"%sresolution_result.txt",out_dir);
        FILE*fp = fopen(report,"w");
        init_report(fp,refs.num_refs);
        for(int r=0;r<refs.num_refs;r++) {
            printf("\tClass %2d",r+1); fflush(stdout);
            Mrc::read(c_mask ,N,N,N,refs[r].mask);
            Mrc::read(c_vol_a,N,N,N,refs[r].h1  );
            Mrc::read(c_vol_b,N,N,N,refs[r].h2  );
            load_map(vol_a,c_vol_a);
            load_map(vol_b,c_vol_b);
            calc_radial(rad_num  ,vol_a,vol_b);
            calc_radial(rad_den_a,vol_a,vol_a);
            calc_radial(rad_den_b,vol_b,vol_b);
            download_fsc(fsc_unmasked);
            Math::mul(c_vol_a,c_mask,N*N*N);
            Math::mul(c_vol_b,c_mask,N*N*N);
            load_map(vol_a,c_vol_a);
            load_map(vol_b,c_vol_b);
            calc_radial(rad_num  ,vol_a,vol_b);
            calc_radial(rad_den_a,vol_a,vol_a);
            calc_radial(rad_den_b,vol_b,vol_b);
            download_fsc(fsc_masked);
            randomize_phase(vol_a,rand_ang_a);
            randomize_phase(vol_b,rand_ang_b);
            calc_radial(rad_num  ,vol_a,vol_b);
            calc_radial(rad_den_a,vol_a,vol_a);
            calc_radial(rad_den_b,vol_b,vol_b);
            download_fsc(fsc_randomized);
            int fpix=0;
            float res = get_resolution(fpix,fsc_masked);
            printf(": %6.3f angstroms.\n",res);
            report_resolution(fp,r,fpix);
            if( save_svg )
                save_svg_fsc(r,res);
        }
        fclose(fp);
    }

protected:

    void setup_gpu() {
        GPU::set_device(gpu_ix);
        real_buffer.alloc(N*N*N);
        vol_a.alloc(M*N*N);
        vol_b.alloc(M*N*N);
        rad_num.alloc(M);
        rad_den_a.alloc(M);
        rad_den_b.alloc(M);
        fft3.alloc(N);

        rand_ang_a.alloc(M*N*N);
        rand_ang_b.alloc(M*N*N);
        float *ptr = new float[M*N*N];
        Math::rand(ptr,M*N*N,2*M_PI);
        cudaMemcpy((void*)rand_ang_a.ptr,(const void*)ptr,sizeof(single)*M*N*N,cudaMemcpyHostToDevice);
        Math::rand(ptr,M*N*N,2*M_PI);
        cudaMemcpy((void*)rand_ang_b.ptr,(const void*)ptr,sizeof(single)*M*N*N,cudaMemcpyHostToDevice);
        delete [] ptr;
    }

    void load_map(GPU::GArrSingle2&vol_out,const float*vol_in) {
        dim3 blk = GPU::get_block_size_2D();
        dim3 grdR = GPU::calc_grid_size(blk,N,N,N);
        dim3 grdC = GPU::calc_grid_size(blk,M,N,N);
        cudaMemcpy((void*)real_buffer.ptr,(const void*)vol_in,sizeof(single)*N*N*N,cudaMemcpyHostToDevice);
        GpuKernels::fftshift3D<<<grdR,blk>>>(real_buffer.ptr,N);
        fft3.exec(vol_out.ptr,real_buffer.ptr);
        GpuKernels::fftshift3D<<<grdC,blk>>>(vol_out.ptr,M,N);
    }

    void calc_radial(GPU::GArrSingle&rad_acc,GPU::GArrSingle2&v_a,GPU::GArrSingle2&v_b) {
        rad_acc.clear();
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,M,N,N);
        GpuKernelsVol::radial_cc<<<grd,blk>>>(rad_acc.ptr,v_a.ptr,v_b.ptr,M,N,1.0/((float)N*N*N));
    }

    void download_fsc(float*fsc_out) {
        dim3 blk;
        dim3 grd;
        blk.x = 1024;
        blk.y = 1;
        blk.z = 1;
        grd.x = GPU::div_round_up(M,1024);
        grd.y = 1;
        grd.z = 1;
        GpuKernelsVol::calc_fsc<<<grd,blk>>>(rad_num.ptr,rad_den_a.ptr,rad_den_b.ptr,M);
        cudaMemcpy((void*)fsc_out,(const void*)rad_num.ptr,sizeof(single)*M,cudaMemcpyDeviceToHost);
        if( abs(fsc_out[0]+1.0) < 0.0001 )
            fsc_out[0] = 1;
    }

    void randomize_phase(GPU::GArrSingle2&vol_out,GPU::GArrSingle&rand_ang) {
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,M,N,N);
        GpuKernelsVol::randomize_phase<<<grd,blk>>>(vol_out.ptr,rand_ang.ptr,rand_fpix,M,N);
    }

    float get_resolution(int&fpix,const float*fsc) {
        fpix=0;
        for(fpix=1;fpix<N/2;fpix++) {
            if( fsc[fpix] < threshold )
                break;
        }
        float res = fpix;
        res = apix*N/res;
        return res;
    }

    void save_svg_fsc(int ref_ix,float resolution) {
        char filename[SUSAN_FILENAME_LENGTH];
        sprintf(filename,"%sfsc_class%03d.svg",out_dir,ref_ix+1);
        SvgFsc svg_fsc(filename,apix);
        svg_fsc.create_grid(rand_fpix,resolution,threshold,N);
        svg_fsc.create_title(ref_ix+1,resolution);
        svg_fsc.add_masked(fsc_masked,M);
        svg_fsc.add_unmask(fsc_unmasked,M);
        svg_fsc.add_rndmzd(fsc_randomized,M);
        svg_fsc.create_legend();
    }

    void init_report(FILE*fp,int num_refs) {
        fprintf(fp,"num_ref:%d\n",num_refs);
    }

    void report_resolution(FILE*fp,int ref_cix,int bp) {
        fprintf(fp,"## Reference %d\nmax_fpix:%d\n",ref_cix+1,bp);
    }
};

#endif /// REFS_ALIGNER_H




