#ifndef REFINE_CTF_H
#define REFINE_CTF_H

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
#include "points_provider.h"
#include "angles_provider.h"
#include "ref_ali.h"
#include "refine_ctf_args.h"


typedef enum {
        CTF_REFINE=1
} CtfCmd;

class CtfRefBuffer {

public:
    GPU::GHostSingle c_substack;
    GPU::GHostFloat4 c_defocus;
    GPU::GArrSingle  g_substack;
    GPU::GArrSingle4 g_defocus;
    CtfConst ctf_vals;
    Particle ptcl;
    float    apix;
    int      K;

    CtfRefBuffer(int N,int max_k) {
        c_substack.alloc(N*N*max_k);
        c_defocus.alloc(max_k);
        g_substack.alloc(N*N*max_k);
        g_defocus.alloc(max_k);
        K = 0;
    }

    ~CtfRefBuffer() {
    }

};

class CtfRefGpuWorker : public Worker {
	
public:
    int gpu_ix;
    int N;
    int M;
    int max_K;
    DoubleBufferHandler *p_buffer;

    float  apix;
    float2 fpix_range;
    float2 def_range;
    float2 astg_range;

    CtfRefGpuWorker() {
    }

    ~CtfRefGpuWorker() {
    }
	
protected:

    void main() {

        GPU::set_device(gpu_ix);
        int current_cmd;
        GPU::Stream stream;
        stream.configure();

        //GPU::GArrSingle ctf_wgt;
        //ctf_wgt.alloc(MP*NP*max_K);

        GPU::sync();

        while( (current_cmd = worker_cmd->read_command()) >= 0 ) {
            switch(current_cmd) {
                case CTF_REFINE:
                    exec_ctf_refinement(stream);
                    break;
                default:
                    break;
            }
        }

        GPU::sync();
    }

    void exec_ctf_refinement(GPU::Stream&stream) {
        p_buffer->RO_sync();
        while( p_buffer->RO_get_status() > DONE ) {
            if( p_buffer->RO_get_status() == READY ) {
                CtfRefBuffer*ptr = (CtfRefBuffer*)p_buffer->RO_get_buffer();
                get_radial_avg(ptr,stream);
                //update_particle_defocus();
                stream.sync();
            }
            p_buffer->RO_sync();
        }
    }

    void get_radial_avg(CtfRefBuffer*ptr,GPU::Stream&stream) {
    }

    void update_particle_defocus(Particle&ptcl,const M33f&Rot,const Vec3&t,const single cc, const int ref_ix,const float apix) {

        ptcl.ali_cc[ref_ix] = cc;

        if( cc > 0 ) {
            M33f Rprv;
            V3f eu_prv;
            eu_prv(0) = ptcl.ali_eu[ref_ix].x;
            eu_prv(1) = ptcl.ali_eu[ref_ix].y;
            eu_prv(2) = ptcl.ali_eu[ref_ix].z;
            Math::eZYZ_Rmat(Rprv,eu_prv);
            M33f Rnew = Rot*Rprv;
            Math::Rmat_eZYZ(eu_prv,Rnew);
            ptcl.ali_eu[ref_ix].x = eu_prv(0);
            ptcl.ali_eu[ref_ix].y = eu_prv(1);
            ptcl.ali_eu[ref_ix].z = eu_prv(2);

            V3f tp,tt;
            tp(0) = t.x;
            tp(1) = t.y;
            tp(2) = t.z;
            tt = Rnew.transpose()*tp;

            ptcl.ali_t[ref_ix].x = tt(0)*apix;
            ptcl.ali_t[ref_ix].y = tt(1)*apix;
            ptcl.ali_t[ref_ix].z = tt(2)*apix;
        }
    }

};

class CtfRefRdrWorker : public Worker {
	
public:
    RefCtfAli::Info *p_info;
    float           *p_stack;
    ParticlesSubset *p_ptcls;
    Tomogram        *p_tomo;
    int gpu_ix;
    int max_K;
    int N;
    int M;

    SubstackCrop    ss_cropper;

    CtfRefRdrWorker() {
    }

    ~CtfRefRdrWorker() {
    }
	
    void setup_global_data(int id,int in_max_K,RefCtfAli::Info*info,WorkerCommand*in_worker_cmd) {
        worker_id  = id;
        worker_cmd = in_worker_cmd;

        p_info   = info;
        gpu_ix   = info->p_gpu[ id % info->n_threads ];
        max_K    = in_max_K;

        N = info->box_size;
        M = (N/2)+1;
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

        printf("Hi from meh thread: %d\n",gpu_ix);

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
                case CTF_REFINE:
                    crop_loop(stack_buffer,stream);
                    break;
                default:
                    break;
            }
        }
        gpu_worker.wait();
    }

    void init_processing_worker(CtfRefGpuWorker&gpu_worker,DoubleBufferHandler*stack_buffer) {
        float res2pix = ((float)N)*p_tomo->pix_size;
        gpu_worker.worker_id    = worker_id;
        gpu_worker.worker_cmd   = worker_cmd;
        gpu_worker.gpu_ix       = gpu_ix;
        gpu_worker.p_buffer     = stack_buffer;
        gpu_worker.N            = N;
        gpu_worker.M            = M;
        gpu_worker.max_K        = max_K;
        gpu_worker.apix         = p_tomo->pix_size;
        gpu_worker.fpix_range.x = floor(res2pix/p_info->res_min);
        gpu_worker.fpix_range.y = floor(res2pix/p_info->res_max);
        gpu_worker.def_range.x  = p_info->def_min;
        gpu_worker.def_range.y  = p_info->def_max;
        gpu_worker.astg_range.x = p_info->astg_ang;
        gpu_worker.astg_range.y = p_info->astg_def;
        gpu_worker.start();
    }

    void crop_loop(DoubleBufferHandler&stack_buffer,GPU::Stream&stream) {
        stack_buffer.WO_sync(EMPTY);
        for(int i=worker_id;i<p_ptcls->n_ptcl;i+=p_info->n_threads) {
            work_progress++;
            CtfRefBuffer*ptr = (CtfRefBuffer*)stack_buffer.WO_get_buffer();
            p_ptcls->get(ptr->ptcl,i);
            read_defocus(ptr);
            crop_substack(ptr);
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

        for(int k=0;k<ptr->K;k++) {
            ptr->c_defocus.ptr[k].x = ptr->ptcl.def[k].U;
            ptr->c_defocus.ptr[k].y = ptr->ptcl.def[k].V;
            ptr->c_defocus.ptr[k].z = ptr->ptcl.def[k].angle;
            ptr->c_defocus.ptr[k].w = ptr->ptcl.prj_w[k];
        }
    }

    void crop_substack(CtfRefBuffer*ptr) {
        V3f pt_tomo,pt_stack,pt_crop;

        /// P_tomo = P_ptcl + t_ali
        pt_tomo(0) = ptr->ptcl.pos().x;
        pt_tomo(1) = ptr->ptcl.pos().y;
        pt_tomo(2) = ptr->ptcl.pos().z;

        for(int k=0;k<ptr->K;k++) {
            if( ptr->ptcl.prj_w[k] > 0 ) {

                /// P_stack = R^k_tomo*P_tomo + t^k_tomo
                pt_stack = p_tomo->R[k]*pt_tomo + p_tomo->t[k];

                /// Angstroms -> pixels
                pt_crop = pt_stack/p_tomo->pix_size + p_tomo->stk_center;

                /// Crop
                if( ss_cropper.check_point(pt_crop) ) {
                    ss_cropper.crop(ptr->c_substack.ptr,p_stack,pt_crop,k);
                    float avg,std;
                    Math::get_avg_std(avg,std,ptr->c_substack.ptr,N*N);
                    if( std > SUSAN_FLOAT_TOL )
                        ss_cropper.normalize_zero_mean_one_std(ptr->c_substack.ptr,k);
                    else
                        ptr->c_defocus.ptr[k].w = 0;

                }
                else {
                    ptr->c_defocus.ptr[k].w = 0;
                }
            }
            else {
                ptr->c_defocus.ptr[k].w = 0;
            }
        }
    }

    bool check_substack(CtfRefBuffer*ptr) {
        bool rslt = false;
        for(int k=0;k<ptr->K;k++) {
            if( ptr->c_defocus.ptr[k].w > 0 )
                rslt = true;
        }
        return rslt;
    }

    void upload(CtfRefBuffer*ptr,cudaStream_t&strm) {
        GPU::upload_async(ptr->g_substack.ptr,ptr->c_substack.ptr,N*N*max_K,strm);
        GPU::upload_async(ptr->g_defocus.ptr,ptr->c_defocus.ptr,max_K    ,strm);
    }

};

class CtfRefPool : public PoolCoordinator {

public:
    CtfRefRdrWorker *workers;
    RefCtfAli::Info *p_info;
    WorkerCommand w_cmd;
    int max_K;
    int N;
    int M;
    int n_ptcls;

    Math::Timing timer;

    char progress_buffer[68];
    char progress_clear [69];

    CtfRefPool(RefCtfAli::Info*info,int in_max_K,int num_ptcls,StackReader&stkrdr,int in_num_threads)
     : PoolCoordinator(stkrdr,in_num_threads), w_cmd(2*in_num_threads+1)
    {
        workers  = new CtfRefRdrWorker[in_num_threads];
        p_info   = info;
        max_K    = in_max_K;
        n_ptcls  = num_ptcls;
        N = info->box_size;
        M = (N/2)+1;
        printf("Hi from main thread\n");
    }

    ~CtfRefPool() {
        delete [] workers;
    }
	
protected:
    void coord_init() {
        for(int i=0;i<p_info->n_threads;i++) {
            workers[i].setup_global_data(i,max_K,p_info,&w_cmd);
            workers[i].start();
        }
    }

    void coord_main(float*stack,ParticlesSubset&ptcls,Tomogram&tomo) {

        int count;

        printf("        Processing: tomo %d with %d particles: %6.2f%%",tomo.tomo_id,ptcls.n_ptcl,0);
        fflush(stdout);

        w_cmd.presend_sync();
        for(int i=0;i<p_info->n_threads;i++) {
            workers[i].setup_working_data(stack,&ptcls,&tomo);
        }
        w_cmd.send_command(CtfCmd::CTF_REFINE);


        while( (count=count_progress()) < ptcls.n_ptcl ) {
            printf("\b\b\b\b\b\b\b%6.2f%%",100*float(count)/float(ptcls.n_ptcl));
            fflush(stdout);
            sleep(1);
        }
        printf("\b\b\b\b\b\b\b100.00%%\n"); fflush(stdout);

        w_cmd.presend_sync();
        w_cmd.send_command(WorkerCommand::BasicCommands::CMD_IDLE);

    }
	
    void coord_end() {
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

};

#endif /// ALIGNER_H


