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

#ifndef ESTIMATE_CTF_H
#define ESTIMATE_CTF_H

#include <iostream>
#include "datatypes.h"
#include "thread_sharing.h"
#include "thread_base.h"
#include "pool_coordinator.h"
#include "particles.h"
#include "tomogram.h"
#include "stack_reader.h"
#include "estimate_ctf_args.h"
#include "gpu.h"
#include "gpu_kernel.h"
#include "gpu_kernel_ctf.h"
#include "substack_crop.h"
#include "mrc.h"
#include "io.h"
#include "points_provider.h"
#include "svg.h"
#include "ctf_normalizer.h"
#include "ctf_linearizer.h"

typedef enum {
	CTF_AVG=1
} CtfCmd;

class CtfBuffer {
	
public:
    GPU::GHostSingle c_substack;
    GPU::GHostFloat2 c_pi_lambda_dZ;
    GPU::GArrSingle  g_substack;
    GPU::GArrSingle2 g_pi_lambda_dZ;
    Particle ptcl;
    float    apix;
    int      K;

    CtfBuffer(int N,int max_k) {
        c_substack.alloc(N*N*max_k);
        g_substack.alloc(N*N*max_k);
        c_pi_lambda_dZ.alloc(max_k);
        g_pi_lambda_dZ.alloc(max_k);
        K = 0;
    }

    ~CtfBuffer() {
    }
	
};

class CtfGpuWorker : public Worker {
	
public:
	int gpu_ix;
	int N;
	int max_K;
	float binning;
	DoubleBufferHandler *p_buffer;
	GPU::GHostSingle    *c_rslt;
	
	CtfGpuWorker() {
	}
	
	~CtfGpuWorker() {
	}
	
protected:
	void main() {
		
		GPU::set_device(gpu_ix);
		CtfNormalizer normalizer(N,max_K);
		int current_cmd;
		
		while( (current_cmd = worker_cmd->read_command()) >= 0 ) {
			switch(current_cmd) {
				case CTF_AVG:
					ctf_avg_ps(normalizer);
					break;
				default:
					break;
			}
		}		
	}
	
	void ctf_avg_ps(CtfNormalizer&gpu) {
		p_buffer->RO_sync();
		gpu.clear_acc();
		while( p_buffer->RO_get_status() > DONE ) {
			if( p_buffer->RO_get_status() == READY ) {
				CtfBuffer*ptr = (CtfBuffer*)p_buffer->RO_get_buffer();
				gpu.process(ptr->g_substack.ptr,ptr->g_pi_lambda_dZ.ptr,ptr->apix,binning);
			}
			p_buffer->RO_sync();
		}
		gpu.apply_wgt();
		gpu.download(c_rslt->ptr);
		gpu.stream.sync();
	}
	
};

class CtfRdrWorker : public Worker {
	
public:
	ArgsCTF::Info   *p_info;
	float           *p_stack;
	ParticlesSubset *p_ptcls;
	Tomogram        *p_tomo;
	int             gpu_ix;
	int             max_K;
	
	GPU::GHostSingle c_rslt;
	
	SubstackCrop    ss_cropper;

	CtfRdrWorker() {
	}
	
	~CtfRdrWorker() {
	}
	
	void setup_global_data(int id,int in_max_K,ArgsCTF::Info*info,WorkerCommand*in_worker_cmd) {
		worker_id  = id;
		worker_cmd = in_worker_cmd;
		p_info     = info;
		gpu_ix     = info->p_gpu[ id % info->n_threads ];
		max_K      = in_max_K;
	}
	
	void setup_working_data(float*stack,ParticlesSubset*ptcls,Tomogram*tomo) {
		p_stack = stack;
		p_ptcls = ptcls;
		p_tomo  = tomo;
		ss_cropper.setup(tomo,p_info->box_size);
		work_progress = 0;
	}
	
protected:
	void main() {
		
		GPU::set_device(gpu_ix);
		GPU::Stream stream;
		stream.configure();
		c_rslt.alloc(((p_info->box_size/2)+1)*p_info->box_size*max_K);
		CtfBuffer buffer_a(p_info->box_size,max_K);
		CtfBuffer buffer_b(p_info->box_size,max_K);
		PBarrier local_barrier(2);
		DoubleBufferHandler stack_buffer((void*)&buffer_a,(void*)&buffer_b,&local_barrier);
		
		CtfGpuWorker gpu_worker;
		init_processing_worker(gpu_worker,&stack_buffer);
		
		int current_cmd;
		
		while( (current_cmd = worker_cmd->read_command()) >= 0 ) {
			switch(current_cmd) {
				case CTF_AVG:
					ctf_avg_ps(stack_buffer,stream);
					break;
				default:
					break;
			}
		}
		gpu_worker.wait();
	}
	
	void init_processing_worker(CtfGpuWorker&gpu_worker,DoubleBufferHandler*stack_buffer) {
		gpu_worker.worker_id  = worker_id;
		gpu_worker.worker_cmd = worker_cmd;
		gpu_worker.gpu_ix     = gpu_ix;
		gpu_worker.p_buffer   = stack_buffer;
		gpu_worker.N          = p_info->box_size;
		gpu_worker.max_K      = max_K;
		gpu_worker.c_rslt     = &c_rslt;
		gpu_worker.binning    = pow(2.0,p_info->binning);
		gpu_worker.start();
	}
	
	void ctf_avg_ps(DoubleBufferHandler&stack_buffer,GPU::Stream&stream) {
		work_progress = 0;
		stack_buffer.WO_sync(EMPTY);
		float pi_lambda = M_PI*Math::get_lambda( p_tomo->KV );
		for(int i=worker_id;i<p_ptcls->n_ptcl;i+=p_info->n_threads) {
			CtfBuffer*ptr = (CtfBuffer*)stack_buffer.WO_get_buffer();
			p_ptcls->get(ptr->ptcl,i);
			ptr->apix = p_tomo->pix_size;
			ptr->K    = p_tomo->stk_dim.z;
			crop_substack(ptr->c_substack.ptr,ptr->c_pi_lambda_dZ.ptr,pi_lambda,ptr->ptcl.pos(),ptr->K);
			work_progress++;
			if( check_substack(ptr->c_pi_lambda_dZ.ptr,ptr->K) ) {
				GPU::upload_async(ptr->g_substack.ptr,ptr->c_substack.ptr,p_info->box_size*p_info->box_size*ptr->K,stream.strm);
				GPU::upload_async(ptr->g_pi_lambda_dZ.ptr,ptr->c_pi_lambda_dZ.ptr,ptr->K,stream.strm);
				stream.sync();
				stack_buffer.WO_sync(READY);
			}
		}
		stack_buffer.WO_sync(DONE);
	}
	
	void crop_substack(single*p_substack,float2*p_dZ,const float pi_lambda,Vec3&pt_ptcl,int K) {
		V3f pt_tomo,pt_stack,pt_crop,pt_subpix,eu_ZYZ;
		
		pt_tomo(0) = pt_ptcl.x;
		pt_tomo(1) = pt_ptcl.y;
		pt_tomo(2) = pt_ptcl.z;
		
		for(int k=0;k<K;k++) {
			/// P_crop = R^k_tomo*P_tomo + t^k_tomo
			pt_stack = p_tomo->R[k]*pt_tomo + p_tomo->t[k];

			/// Angstroms -> pixels
			pt_crop = pt_stack/p_tomo->pix_size + p_tomo->stk_center;

			/// Setup data for upload to GPU
			p_dZ[k].x = pi_lambda*pt_stack(2);
			p_dZ[k].y = 1;
			
			/// Crop
			if( ss_cropper.check_point(pt_crop) ) {
				ss_cropper.crop(p_substack,p_stack,pt_crop,k);
				ss_cropper.normalize_zero_mean_one_std(p_substack,k);
			}
			else {
				p_dZ[k].x = 0;
				p_dZ[k].y = 0;
			}
		}
	}
	
	bool check_substack(float2*p_dZ,int K) {
		bool rslt = true;
		for(int k=0;k<K;k++) {
			if( p_dZ[k].y < 1 )
				rslt = false;
		}
		return rslt;
	}
};

class CtfEstimatePool : public PoolCoordinator {

public:
	CtfRdrWorker *workers;
	ArgsCTF::Info *p_info;
	WorkerCommand w_cmd;
	int total_threads;
	int max_K;

	CtfEstimatePool(ArgsCTF::Info*info,int in_max_K,StackReader&stkrdr,int in_num_threads)
	 : PoolCoordinator(stkrdr,in_num_threads), w_cmd(2*in_num_threads+1) {
		workers = new CtfRdrWorker[in_num_threads];
		p_info  = info;
		max_K = in_max_K;
	}
	
	~CtfEstimatePool() {
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
		char filename[SUSAN_FILENAME_LENGTH];
		sprintf(filename,"%s/Tomo%03d",p_info->out_dir,tomo.tomo_id);
		IO::create_dir(filename);
		
		printf("        Processing: tomo %d with %d particles: %6.2f%%",tomo.tomo_id,ptcls.n_ptcl,0);
		fflush(stdout);
		
		w_cmd.presend_sync();
		for(int i=0;i<p_info->n_threads;i++) {
			workers[i].setup_working_data(stack,&ptcls,&tomo);
		}
		w_cmd.send_command(CtfCmd::CTF_AVG);
		
		while( (count=count_progress()) < ptcls.n_ptcl ) {
			printf("\b\b\b\b\b\b\b%6.2f%%",100*float(count)/float(ptcls.n_ptcl));
			fflush(stdout);
			sleep(1);
		}
		printf("\b\b\b\b\b\b\b100.00%%\n"); fflush(stdout);
		
		w_cmd.presend_sync();
		clear_workers();
		reduce_and_bcast();
		if( p_info->verbose > 0 ) {
			sprintf(filename,"%s/Tomo%03d/ctf_normalized_raw.mrc",p_info->out_dir,tomo.tomo_id);
			Mrc::write(workers[0].c_rslt.ptr,(p_info->box_size/2)+1,p_info->box_size,tomo.num_proj,filename);
		}
		sprintf(filename,"%s/Tomo%03d",p_info->out_dir,tomo.tomo_id);
		post_process(filename,workers[0].c_rslt.ptr,tomo);
		w_cmd.send_command(WorkerCommand::BasicCommands::CMD_IDLE);
		
	}
	
	void coord_end() {
		w_cmd.send_command(WorkerCommand::BasicCommands::CMD_END);
		for(int i=0;i<p_info->n_threads;i++) {
			workers[i].wait();
		}
	}
	
	int count_progress() {
		int count = 0;
		for(int i=0;i<p_info->n_threads;i++) {
			count += workers[i].work_progress;
		}
		return count;
	}

	void reduce_and_bcast() {
		int l = (p_info->box_size/2+1)*p_info->box_size*max_K;
		if( p_info->n_threads > 1 ) {
			for(int i=1;i<p_info->n_threads;i++)
				Math::sum(workers[0].c_rslt.ptr,workers[i].c_rslt.ptr,l);
			
			for(int i=1;i<p_info->n_threads;i++)
				memcpy(workers[i].c_rslt.ptr,workers[0].c_rslt.ptr,l);
		}
	}
	
	void clear_workers() {
		for(int i=1;i<p_info->n_threads;i++)
			workers[i].work_progress = 0;
	}

	void post_process(const char*out_dir,single*ptr,Tomogram&tomo) {
		CtfLinearizer ctf_lin(p_info->p_gpu[0],p_info->box_size,tomo.num_proj);
		ctf_lin.load_info(p_info,&tomo);
		ctf_lin.process(out_dir,ptr,&tomo);
	}

};

#endif /// ESTIMATE_CTF_H


