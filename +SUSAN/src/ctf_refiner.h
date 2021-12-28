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

#ifndef CTF_REFINER_H
#define CTF_REFINER_H

#include <iostream>
#include "datatypes.h"
#include "thread_sharing.h"
#include "thread_base.h"
#include "pool_coordinator.h"
#include "particles.h"
#include "tomogram.h"
#include "reference.h"
#include "ref_maps.h"
#include "stack_reader.h"
#include "gpu.h"
#include "gpu_kernel.h"
#include "gpu_rand.h"
#include "gpu_kernel_ctf.h"
#include "substack_crop.h"
#include "mrc.h"
#include "io.h"
#include "ref_ali.h"
#include "ctf_refiner_args.h"

typedef enum {
    CTF_REF=1
} CtfRefCmd;

class CtfRefBuffer {

public:
    GPU::GHostSingle  c_stk;
    GPU::GHostFloat2  c_pad;
    GPU::GHostProj2D  c_ali;
    GPU::GHostDefocus c_def;
    GPU::GArrSingle   g_stk;
    GPU::GArrSingle2  g_pad;
    GPU::GArrProj2D   g_ali;
    GPU::GArrDefocus  g_def;
    Particle ptcl;
    CtfConst ctf_vals;
    int      K;
    int      r_ix;
    int      class_ix;

    CtfRefBuffer(int N,int max_k) {
        c_stk.alloc(N*N*max_k);
        g_stk.alloc(N*N*max_k);
        c_pad.alloc(max_k);
        g_pad.alloc(max_k);
        c_ali.alloc(max_k);
        g_ali.alloc(max_k);
        c_def.alloc(max_k);
        g_def.alloc(max_k);
        K = 0;
    }

    ~CtfRefBuffer() {
    }

};

class CtfRefGpuWorker : public Worker {

protected:
    class Refine{
    public:
        int M;
        int N;
        int max_K;

        GPU::GArrSingle2 prj_buf;
        GPU::GArrSingle  ctf_ite;

        Refine(int m, int n, int k) {
            M = m;
            N = n;
            max_K = k;
            prj_buf.alloc(m*n*k);
            ctf_ite.alloc(m*n*k);
        }

        ~Refine() {
        }

        void load_buf(GPU::GArrSingle2&data_in,int k,GPU::Stream&stream) {
            GPU::copy_async(prj_buf.ptr,data_in.ptr,M*N*k,stream.strm);
        }

        void apply_ctf(GPU::GArrSingle2&data_out,float3 delta,CtfRefBuffer*ptr,GPU::Stream&stream) {
            int3 ss = make_int3(M,N,ptr->K);
            dim3 blk = GPU::get_block_size_2D();
            dim3 grd = GPU::calc_grid_size(blk,M,N,ptr->K);
            GpuKernelsCtf::create_ctf<<<grd,blk,0,stream.strm>>>(ctf_ite.ptr,delta,ptr->ctf_vals,ptr->g_def.ptr,ss);
            GpuKernels::multiply<<<grd,blk,0,stream.strm>>>(data_out.ptr,ctf_ite.ptr,ss);
        }
    };

public:
    int gpu_ix;
    int N;
    int M;
    int P;
    int R;
    int pad_type;
    int max_K;
    float delta_def;
    float delta_ang;
    bool  est_dose;
    bool  use_halves;
    float3  bandpass;

    DoubleBufferHandler *p_buffer;
    RefMap              *p_refs;

    CtfRefGpuWorker() {
    }

    ~CtfRefGpuWorker() {
    }

protected:
    int NP;
    int MP;

    void main() {

        NP = N+P;
        MP = (NP/2)+1;

        GPU::set_device(gpu_ix);
        int current_cmd;
        GPU::Stream stream;
        stream.configure();

        AliSubstack ss_data(M,N,max_K,P,stream);

        uint32 off_type = ArgsAli::OffsetType_t::CIRCLE;
        float4 off_par = {0,0,0,1};

        AliData ali_data(MP,NP,max_K,off_par,off_type,stream);

        RadialAverager rad_avgr(M,N,max_K);

        Refine ctf_ref(MP,NP,max_K);

        int num_vols = R;
        if( use_halves ) num_vols = 2*R;
        AliRef*vols = new AliRef[num_vols];
        allocate_references(vols);

        GPU::sync();

        while( (current_cmd = worker_cmd->read_command()) >= 0 ) {
            switch(current_cmd) {
                case CTF_REF:
                    refine_ctf(vols,ctf_ref,ss_data,ali_data,rad_avgr,stream);
                    break;
                default:
                    break;
            }
        }

        GPU::sync();
        delete [] vols;
    }

    void allocate_references(AliRef*vols) {
        GPU::GArrSingle  g_raw;
        GPU::GArrSingle  g_pad;
        GPU::GArrSingle2 g_fou;

        g_raw.alloc(N*N*N);
        g_pad.alloc(NP*NP*NP);
        g_fou.alloc(MP*NP*NP);

        GpuFFT::FFT3D fft3;
        fft3.alloc(NP);

        if( use_halves ) {
            for(int r=0;r<R;r++) {
                upload_ref(g_pad,g_raw,p_refs[r].half_A);
                exec_fft3(g_fou,g_pad,fft3);
                vols[2*r  ].allocate(g_fou,MP,NP);

                upload_ref(g_pad,g_raw,p_refs[r].half_B);
                exec_fft3(g_fou,g_pad,fft3);
                vols[2*r+1].allocate(g_fou,MP,NP);
            }
        }
        else {
            for(int r=0;r<R;r++) {
                upload_ref(g_pad,g_raw,p_refs[r].map);
                exec_fft3(g_fou,g_pad,fft3);
                vols[r].allocate(g_fou,MP,NP);
            }
        }
    }

    void upload_ref(GPU::GArrSingle&g_pad,GPU::GArrSingle&g_raw,single*data) {
        cudaError_t err = cudaMemcpy((void*)g_raw.ptr,(const void*)data,sizeof(single)*N*N*N,cudaMemcpyHostToDevice);
        if( err != cudaSuccess ) {
            fprintf(stderr,"Error uploading volume to CUDA memory. ");
            fprintf(stderr,"GPU error: %s.\n",cudaGetErrorString(err));
            exit(1);
        }
        g_pad.clear();

        int3 pad = make_int3(P/2,P/2,P/2);
        int3 ss_raw = make_int3(N,N,N);
        int3 ss_pad = make_int3(NP,NP,NP);

        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,N,N,N);

        GpuKernels::load_pad<<<grd,blk>>>(g_pad.ptr,g_raw.ptr,pad,ss_raw,ss_pad);
    }

    void exec_fft3(GPU::GArrSingle2&g_fou,GPU::GArrSingle&g_pad,GpuFFT::FFT3D&fft3) {
        dim3 blk = GPU::get_block_size_2D();
        dim3 grdR = GPU::calc_grid_size(blk,NP,NP,NP);
        dim3 grdC = GPU::calc_grid_size(blk,MP,NP,NP);
        GpuKernels::fftshift3D<<<grdR,blk>>>(g_pad.ptr,NP);
        fft3.exec(g_fou.ptr,g_pad.ptr);
        GpuKernels::fftshift3D<<<grdC,blk>>>(g_fou.ptr,MP,NP);
    }

    void refine_ctf(AliRef*vols,Refine&ctf_ref,AliSubstack&ss_data,AliData&ali_data,RadialAverager&rad_avgr,GPU::Stream&stream) {
        p_buffer->RO_sync();
        while( p_buffer->RO_get_status() > DONE ) {
            if( p_buffer->RO_get_status() == READY ) {
                CtfRefBuffer*ptr = (CtfRefBuffer*)p_buffer->RO_get_buffer();
                add_data(ss_data,ptr,rad_avgr,stream);
                search_ctf(vols[ptr->r_ix],ss_data,ctf_ref,ptr,ali_data,rad_avgr,stream);
                stream.sync();
            }
            p_buffer->RO_sync();
        }
    }

    void add_data(AliSubstack&ss_data,CtfRefBuffer*ptr,RadialAverager&rad_avgr,GPU::Stream&stream) {

        switch( pad_type ) {
            case ArgsAli::PaddingType_t::PAD_ZERO:
                ss_data.pad_zero(stream);
                break;
            case ArgsAli::PaddingType_t::PAD_GAUSSIAN:
                ss_data.pad_normal(ptr->g_pad,ptr->K,stream);
                break;
            default:
                break;
        }

        ss_data.add_data(ptr->g_stk,ptr->g_ali,ptr->K,stream);

        //static bool flag = true;
        //if( flag ) {
        //    flag = false;
        //if( ptr->ptcl.ptcl_id() == 2122 ) {
        /*if( ptr->ptcl.ptcl_id() == 1 ) {
            stream.sync();
            float*tmp = new float[NP*NP*ptr->K];
            GPU::download_async(tmp,ss_data.ss_padded.ptr,NP*NP*ptr->K,stream.strm);
            stream.sync();
            char filename[SUSAN_FILENAME_LENGTH];
            sprintf(filename,"data_%02d.mrc",ptr->r_ix);
            Mrc::write(tmp,NP,NP,ptr->K,filename);
            delete [] tmp;
        }*/

        rad_avgr.preset_FRC(ss_data.ss_fourier,ptr->K,stream);
    }

    void search_ctf(AliRef&vol,AliSubstack&ss_data,Refine&ctf_ref,CtfRefBuffer*ptr,AliData&ali_data,RadialAverager&rad_avgr,GPU::Stream&stream) {

        single max_cc[ptr->K];
        single ite_cc[ptr->K];
        int ite_idx[ptr->K];
        Defocus def_rslt[ptr->K];

        memset(max_cc,0,sizeof(single)*ptr->K);

        Rot33 Rot;
        M33f  R_eye = Eigen::MatrixXf::Identity(3,3);

        Math::set(Rot,R_eye);
        ali_data.rotate_post(Rot,ptr->g_ali,ptr->K,stream);

        /*if( ptr->ptcl.ptcl_id() == 1 ) {
            ali_data.project(vol.ref,bandpass,ptr->K,stream);
            //ali_data.multiply(ctf_wgt,ptr->K,stream);
            ali_data.invert_fourier(ptr->K,stream);
            float *tmp = new float[NP*NP*ptr->K];
            GPU::download_async(tmp,ali_data.prj_r.ptr,NP*NP*ptr->K,stream.strm);
            stream.sync();
            char fn[SUSAN_FILENAME_LENGTH];
            sprintf(fn,"proj_3D_%d_%d.mrc",ptr->class_ix,ptr->r_ix);
            Mrc::write(tmp,NP,NP,ptr->K,fn);
            delete [] tmp;
        }*/

        ali_data.project(vol.ref,bandpass,ptr->K,stream);
        ctf_ref.load_buf(ali_data.prj_c,ptr->K,stream);

        float dU,dV,dA;
        float3 delta_w;
        for( dU=-delta_def; dU<delta_def; dU+=10 ) {
            delta_w.x = dU;
            for( dV=-delta_def; dV<delta_def; dV+=10 ) {
                delta_w.y = dV;
                for( dA=-delta_ang; dU<delta_ang; dU+=0.5 ) {
                    delta_w.z = dA;
                    ctf_ref.apply_ctf(ali_data.prj_c,delta_w,ptr,stream);
                    rad_avgr.preset_FRC(ali_data.prj_c,ptr->K,stream);
                    ali_data.multiply(ss_data.ss_fourier,ptr->K,stream);
                    ali_data.invert_fourier(ptr->K,stream);
                    stream.sync();
                    ali_data.extract_cc(ite_cc,ite_idx,ptr->K,stream);
                    for(int i=0;i<ptr->K;i++) {
                        if( ite_cc[i] > max_cc[i] ) {
                            max_cc[i] = ite_cc[i];
                            def_rslt[i].U = dU;
                            def_rslt[i].V = dV;
                            def_rslt[i].angle = dA;
                        }
                    }
                }
            }
        }
        update_particle(ptr->ptcl,def_rslt,max_cc,ptr->K);
    }

    void update_particle(Particle&ptcl,const Defocus*delta_def,const single*cc, const int k) {

        for(int i=0;i<k;i++) {
            if( ptcl.prj_w[i] > 0 ) {
                ptcl.def[i].U += delta_def[i].U;
                ptcl.def[i].V += delta_def[i].V;
                ptcl.def[i].angle += delta_def[i].angle;
                if( est_dose )
                    ptcl.def[i].ExpFilt = delta_def[i].ExpFilt;
                ptcl.prj_cc[i] = cc[i];
            }
        }
    }
};

class CtfRefRdrWorker : public Worker {

public:
    ArgsCtfRef::Info *p_info;
    float            *p_stack;
    ParticlesSubset  *p_ptcls;
    Tomogram         *p_tomo;
    RefMap           *p_refs;
    int gpu_ix;
    int max_K;
    int N;
    int M;
    int R;
    int P;
    int pad_type;
    int NP;
    int MP;

    float bp_pad;

    SubstackCrop    ss_cropper;

    CtfRefRdrWorker() {
    }

    ~CtfRefRdrWorker() {
    }

    void setup_global_data(int id,RefMap*in_p_refs,int in_R,int in_max_K,ArgsCtfRef::Info*info,WorkerCommand*in_worker_cmd) {
        worker_id  = id;
        worker_cmd = in_worker_cmd;

        p_info   = info;
        gpu_ix   = info->p_gpu[ id % info->n_threads ];
        max_K    = in_max_K;
        pad_type = info->pad_type;

        N = info->box_size;
        M = (N/2)+1;
        P = info->pad_size;

        NP = N + P;
        MP = (NP/2)+1;

        p_refs = in_p_refs;
        R = in_R;
    }

    void setup_working_data(float*stack,ParticlesSubset*ptcls,Tomogram*tomo) {
        p_stack = stack;
        p_ptcls = ptcls;
        p_tomo  = tomo;
        ss_cropper.setup(tomo,N);
        work_progress=0;
    }

protected:
    void main() {

        GPU::set_device(gpu_ix);
        GPU::Stream stream;
        stream.configure();
        work_accumul = 0;
        CtfRefBuffer buffer_a(N,max_K);
        CtfRefBuffer buffer_b(N,max_K);
        PBarrier local_barrier(2);
        DoubleBufferHandler stack_buffer((void*)&buffer_a,(void*)&buffer_b,&local_barrier);

        CtfRefGpuWorker gpu_worker;
        init_processing_worker(gpu_worker,&stack_buffer);

        int current_cmd;

        while( (current_cmd = worker_cmd->read_command()) >= 0 ) {
            switch(current_cmd) {
                case CTF_REF:
                    crop_loop(stack_buffer,stream);
                    break;
                default:
                    break;
            }
        }
        gpu_worker.wait();
    }

    void init_processing_worker(CtfRefGpuWorker&gpu_worker,DoubleBufferHandler*stack_buffer) {
        bp_pad = p_info->fpix_roll/2;
        float bp_scale = ((float)NP)/((float)N);
        gpu_worker.worker_id  = worker_id;
        gpu_worker.worker_cmd = worker_cmd;
        gpu_worker.gpu_ix     = gpu_ix;
        gpu_worker.p_buffer   = stack_buffer;
        gpu_worker.N          = N;
        gpu_worker.M          = M;
        gpu_worker.P          = P;
        gpu_worker.R          = R;
        gpu_worker.p_refs     = p_refs;
        gpu_worker.use_halves = p_info->use_halves;
        gpu_worker.pad_type   = pad_type;
        gpu_worker.max_K      = max_K;
        gpu_worker.bandpass.x = max(bp_scale*p_info->fpix_min-bp_pad,0.0);
        gpu_worker.bandpass.y = min(bp_scale*p_info->fpix_max+bp_pad,(float)NP);
        gpu_worker.bandpass.z = sqrt(p_info->fpix_roll);
        gpu_worker.delta_def  = p_info->delta_def;
        gpu_worker.delta_ang  = p_info->delta_ang;
        gpu_worker.est_dose   = p_info->est_dose;
        gpu_worker.start();
    }

    void crop_loop(DoubleBufferHandler&stack_buffer,GPU::Stream&stream) {
        stack_buffer.WO_sync(EMPTY);
        for(int i=worker_id;i<p_ptcls->n_ptcl;i+=p_info->n_threads) {
            work_progress++;
            work_accumul++;
            CtfRefBuffer*ptr = (CtfRefBuffer*)stack_buffer.WO_get_buffer();
            p_ptcls->get(ptr->ptcl,i);
            read_defocus(ptr);
            crop_substack(ptr,ptr->ptcl.ref_cix());
            if( check_substack(ptr) ) {
                upload(ptr,stream.strm);
                stream.sync();
                stack_buffer.WO_sync(READY);
            }
        }
        stack_buffer.WO_sync(DONE);
    }

    void read_defocus(CtfRefBuffer*ptr) {
        ptr->K = p_tomo->stk_dim.z;

        float lambda = Math::get_lambda( p_tomo->KV );

        ptr->ctf_vals.AC = p_tomo->AC;
        ptr->ctf_vals.CA = sqrt(1-p_tomo->AC*p_tomo->AC);
        ptr->ctf_vals.apix = p_tomo->pix_size;
        ptr->ctf_vals.LambdaPi = M_PI*lambda;
        ptr->ctf_vals.CsLambda3PiH = lambda*lambda*lambda*(p_tomo->CS*1e7)*M_PI/2;

        memcpy( (void**)(ptr->c_def.ptr), (const void**)(ptr->ptcl.def), sizeof(Defocus)*ptr->K  );

        for(int k=0;k<ptr->K;k++) {
            if( ptr->c_def.ptr[k].max_res > 0 ) {
                ptr->c_def.ptr[k].max_res = ((float)NP)*p_tomo->pix_size/ptr->c_def.ptr[k].max_res;
                ptr->c_def.ptr[k].max_res = min(ptr->c_def.ptr[k].max_res+bp_pad,(float)NP/2);
            }
        }
    }

    void crop_substack(CtfRefBuffer*ptr,const int r) {

        V3f pt_tomo,pt_stack,pt_crop,pt_subpix,eu_ZYZ;
        M33f R_tmp,R_ali,R_stack,R_gpu;

        if( p_info->use_halves )
            ptr->r_ix = 2*r + (ptr->ptcl.half_id()-1);
        else
            ptr->r_ix = r;

        /// P_tomo = P_ptcl + t_ali
        pt_tomo(0) = ptr->ptcl.pos().x + ptr->ptcl.ali_t[r].x;
        pt_tomo(1) = ptr->ptcl.pos().y + ptr->ptcl.ali_t[r].y;
        pt_tomo(2) = ptr->ptcl.pos().z + ptr->ptcl.ali_t[r].z;

        eu_ZYZ(0) = ptr->ptcl.ali_eu[r].x;
        eu_ZYZ(1) = ptr->ptcl.ali_eu[r].y;
        eu_ZYZ(2) = ptr->ptcl.ali_eu[r].z;
        Math::eZYZ_Rmat(R_ali,eu_ZYZ);

        for(int k=0;k<ptr->K;k++) {
            if( ptr->ptcl.prj_w[k] > 0 ) {

                /// P_stack = R^k_tomo*P_tomo + t^k_tomo
                pt_stack = p_tomo->R[k]*pt_tomo + p_tomo->t[k];

                /// P_crop = R^k_prj*P_stack + t^k_prj
                eu_ZYZ(0) = ptr->ptcl.prj_eu[k].x;
                eu_ZYZ(1) = ptr->ptcl.prj_eu[k].y;
                eu_ZYZ(2) = ptr->ptcl.prj_eu[k].z;
                Math::eZYZ_Rmat(R_tmp,eu_ZYZ);
                //pt_crop = R_tmp*pt_stack;
                pt_crop = pt_stack;
                pt_crop(0) += ptr->ptcl.prj_t[k].x;
                pt_crop(1) += ptr->ptcl.prj_t[k].y;

                /// R_stack = R^k_prj*R^k_tomo
                R_stack = R_tmp*p_tomo->R[k];

                /// Angstroms -> pixels
                pt_crop = pt_crop/p_tomo->pix_size + p_tomo->stk_center;

                /// Get subpixel shift
                V3f pt_tmp;
                pt_tmp(0) = pt_crop(0) - floor(pt_crop(0));
                pt_tmp(1) = pt_crop(1) - floor(pt_crop(1));
                pt_tmp(2) = 0;
                pt_subpix = R_stack.transpose()*pt_tmp;

                /// Setup data for upload to GPU
                ptr->c_ali.ptr[k].t.x = -pt_subpix(0);
                ptr->c_ali.ptr[k].t.y = -pt_subpix(1);
                ptr->c_ali.ptr[k].t.z = 0;
                ptr->c_ali.ptr[k].w = ptr->ptcl.prj_w[k];
                R_gpu = (R_ali)*(R_stack.transpose());
                Math::set( ptr->c_ali.ptr[k].R, R_gpu );

                /// Crop
                if( ss_cropper.check_point(pt_crop) ) {
                    ss_cropper.crop(ptr->c_stk.ptr,p_stack,pt_crop,k);
                    ptr->c_pad.ptr[k].x = 0;
                    ptr->c_pad.ptr[k].y = 1;
                    ss_cropper.normalize_zero_mean_one_std(ptr->c_stk.ptr,k);
                    ptr->ptcl.prj_w[k] = ptr->c_pad.ptr[k].y;
                }
                else {
                    ptr->c_ali.ptr[k].w = 0;
                }
            }
            else {
                ptr->c_ali.ptr[k].w = 0;
            }
        }
    }

    bool check_substack(CtfRefBuffer*ptr) {
        bool rslt = false;
        for(int k=0;k<ptr->K;k++) {
            if( ptr->c_ali.ptr[k].w > 0 )
                rslt = true;
        }
        return rslt;
    }

    void upload(CtfRefBuffer*ptr,cudaStream_t&strm) {
        GPU::upload_async(ptr->g_stk.ptr,ptr->c_stk.ptr,N*N*max_K,strm);
        GPU::upload_async(ptr->g_pad.ptr,ptr->c_pad.ptr,max_K    ,strm);
        GPU::upload_async(ptr->g_ali.ptr,ptr->c_ali.ptr,max_K    ,strm);
        GPU::upload_async(ptr->g_def.ptr,ptr->c_def.ptr,max_K    ,strm);
    }

};

class CtfRefinerPool : public PoolCoordinator {

public:
    CtfRefRdrWorker  *workers;
    ArgsCtfRef::Info *p_info;
    WorkerCommand    w_cmd;
    RefMap           *p_refs;
    int max_K;
    int N;
    int M;
    int R;
    int P;
    int n_ptcls;
    int NP;
    int MP;

    Math::Timing timer;

    char progress_buffer[68];
    char progress_clear [69];

    CtfRefinerPool(ArgsCtfRef::Info*info,References*in_p_refs,int in_max_K,int num_ptcls,StackReader&stkrdr,int in_num_threads)
     : PoolCoordinator(stkrdr,in_num_threads), w_cmd(2*in_num_threads+1)
    {
        workers  = new CtfRefRdrWorker[in_num_threads];
        p_info   = info;
        max_K    = in_max_K;
        n_ptcls  = num_ptcls;
        N = info->box_size;
        M = (N/2)+1;
        P = info->pad_size;
        NP = N+P;
        MP = (NP/2)+1;

        memset(progress_buffer,' ',66);
        memset(progress_clear,'\b',66);
        progress_buffer[66] = 0;
        progress_buffer[67] = 0;
        if( SUSAN_CARRIER_RETURN == '\r' ) {
            progress_clear [66] = SUSAN_CARRIER_RETURN;
        }
        else {
            progress_clear [65] = 0;
        }
        progress_clear [67] = 0;

        load_references(in_p_refs);
    }

    ~CtfRefinerPool() {
        delete [] p_refs;
        delete [] workers;
    }

protected:
    void load_references(References*in_p_refs) {
        R = in_p_refs->num_refs;
        p_refs = new RefMap[R];
        for(int r=0;r<R;r++) {
            p_refs[r].load(in_p_refs->at(r));
        }
    }

    void coord_init() {
        for(int i=0;i<p_info->n_threads;i++) {
            workers[i].setup_global_data(i,p_refs,R,max_K,p_info,&w_cmd);
            workers[i].start();
        }
        progress_start();
    }

    void coord_main(float*stack,ParticlesSubset&ptcls,Tomogram&tomo) {

        w_cmd.presend_sync();
        for(int i=0;i<p_info->n_threads;i++) {
            workers[i].setup_working_data(stack,&ptcls,&tomo);
        }
        w_cmd.send_command(CtfRefCmd::CTF_REF);

        show_progress(ptcls.n_ptcl);

        w_cmd.send_command(WorkerCommand::BasicCommands::CMD_IDLE);

    }

    void coord_end() {
        show_done();
        w_cmd.send_command(WorkerCommand::BasicCommands::CMD_END);
        for(int i=0;i<p_info->n_threads;i++) {
            workers[i].wait();
        }
    }

    long count_progress() {
        long count = 0;
        for(int i=0;i<p_info->n_threads;i++) {
            count += workers[i].work_progress;
        }
        return count;
    }

    long count_accumul() {
        long count = 0;
        for(int i=0;i<p_info->n_threads;i++) {
            count += workers[i].work_accumul;
        }
        return count;
    }

    virtual void progress_start() {
        timer.tic();
        sprintf(progress_buffer,"        Refining particles: Buffering...");
        int n = strlen(progress_buffer);
        progress_buffer[n]  = ' ';
        progress_buffer[65] = 0;
        printf(progress_buffer);
        fflush(stdout);
    }

    virtual void show_progress(const int ptcls_in_tomo) {
        int cur_progress=0;
        while( (cur_progress=count_progress()) < ptcls_in_tomo ) {
            memset(progress_buffer,' ',66);
            if( cur_progress > 0 ) {
                int progress = count_accumul();
                float progress_percent = 100*(float)progress/float(n_ptcls);
                sprintf(progress_buffer,"        Refining particles: %6.2f%%%%",progress_percent);
                int n = strlen(progress_buffer);
                add_etc(progress_buffer+n,progress,n_ptcls);
            }
            else {
                sprintf(progress_buffer,"        Refining particles: Buffering...");
                int n = strlen(progress_buffer);
                progress_buffer[n]  = ' ';
                progress_buffer[65] = 0;
            }
            printf(progress_clear);
            fflush(stdout);
            printf(progress_buffer);
            fflush(stdout);
            sleep(1);
        }
    }

    virtual void show_done() {
        memset(progress_buffer,' ',66);
        sprintf(progress_buffer,"        Refining particles: 100.00%%%%");
        int n = strlen(progress_buffer);
        progress_buffer[n] = ' ';
        printf(progress_clear);
        printf(progress_buffer);
        printf("\n");
        fflush(stdout);
    }

    void add_etc(char*buffer,int progress,int total) {

        int days,hours,mins,secs;

        timer.get_etc(days,hours,mins,secs,progress,total);

        if( days > 0 ) {
            sprintf(buffer," (ETC: %dd %02d:%02d:%02d)",days,hours,mins,secs);
            int n = strlen(buffer);
            buffer[n] = ' ';
        }
        else {
            sprintf(buffer," (ETC: %02d:%02d:%02d)",hours,mins,secs);
            int n = strlen(buffer);
            buffer[n] = ' ';
        }
    }

};

#endif /// CTF_REFINER_H


