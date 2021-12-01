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

#ifndef RECONSTRUCT_H
#define RECONSTRUCT_H

#include <iostream>
#include "datatypes.h"
#include "thread_sharing.h"
#include "thread_base.h"
#include "pool_coordinator.h"
#include "particles.h"
#include "tomogram.h"
#include "stack_reader.h"
#include "gpu.h"
#include "gpu_kernel.h"
#include "gpu_kernel_ctf.h"
#include "substack_crop.h"
#include "mrc.h"
#include "io.h"
#include "points_provider.h"
#include "rec_acc.h"
#include "reconstruct_args.h"

typedef enum {
    REC_EXEC=1
} RecCmd;

class RecBuffer {
	
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

    RecBuffer(int N,int max_k) {
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

    ~RecBuffer() {
    }
	
};

class RecGpuWorker : public Worker {
	
public:
    int gpu_ix;
    int N;
    int M;
    int P;
    int R;
    int pad_type;
    int ctf_type;
    int max_K;
    float3  bandpass;
    float2  ssnr; /// x=F; y=S;
    double2 **c_acc;
    double  **c_wgt;
    DoubleBufferHandler *p_buffer;

    RecGpuWorker() {
    }

    ~RecGpuWorker() {
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
        RecSubstack ss_data(M,N,max_K,P,stream);
        RecAcc vols[R];
        for(int r=0;r<R;r++)
            vols[r].alloc(MP,NP,max_K);

        while( (current_cmd = worker_cmd->read_command()) >= 0 ) {
            switch(current_cmd) {
                case REC_EXEC:
                    insert_loop(vols,ss_data,stream);
                    break;
                default:
                    break;
            }
        }

        for(int r=0;r<R;r++)
            download_vol(c_acc[r],c_wgt[r],vols[r]);

        GPU::sync();
    }
	
    void insert_loop(RecAcc*vols,RecSubstack&ss_data,GPU::Stream&stream) {
        p_buffer->RO_sync();
        while( p_buffer->RO_get_status() > DONE ) {
            if( p_buffer->RO_get_status() == READY ) {
                RecBuffer*ptr = (RecBuffer*)p_buffer->RO_get_buffer();
                add_data(ss_data,ptr,stream);
                correct_ctf(ss_data,ptr,stream);
                insert_vol(vols[ptr->r_ix],ss_data,ptr,stream);
                stream.sync();
            }
            p_buffer->RO_sync();
        }
    }
	
    void add_data(RecSubstack&ss_data,RecBuffer*ptr,GPU::Stream&stream) {
        if( pad_type == ArgsRec::PaddingType_t::PAD_ZERO )
            ss_data.pad_zero(stream);
        if( pad_type == ArgsRec::PaddingType_t::PAD_GAUSSIAN )
            ss_data.pad_normal(ptr->g_pad,ptr->K,stream);

        ss_data.add_data(ptr->g_stk,ptr->g_ali,ptr->K,stream);
    }

    void correct_ctf(RecSubstack&ss_data,RecBuffer*ptr,GPU::Stream&stream) {
        if( ctf_type == ArgsRec::InversionType_t::NO_INV )
            ss_data.set_no_ctf(bandpass,ptr->K,stream);
        if( ctf_type == ArgsRec::InversionType_t::PHASE_FLIP )
            ss_data.set_phase_flip(ptr->ctf_vals,ptr->g_def,bandpass,ptr->K,stream);
        if( ctf_type == ArgsRec::InversionType_t::WIENER_INV )
            ss_data.set_wiener(ptr->ctf_vals,ptr->g_def,bandpass,ptr->K,stream);
        if( ctf_type == ArgsRec::InversionType_t::WIENER_INV_SSNR )
            ss_data.set_wiener_ssnr(ptr->ctf_vals,ptr->g_def,bandpass,ssnr,ptr->K,stream);

    }
	
	void insert_vol(RecAcc&vol,RecSubstack&ss_data,RecBuffer*ptr,GPU::Stream&stream) {
		vol.insert(ss_data.ss_tex,ss_data.ss_ctf,ptr->g_ali,bandpass,ptr->K,stream);
	}
	
	void download_vol(double2*p_acc,double*p_wgt,RecAcc&vol) {
		cudaMemcpy( (void*)(p_acc), (const void*)vol.vol_acc.ptr, sizeof(double2)*NP*NP*MP, cudaMemcpyDeviceToHost);
		cudaMemcpy( (void*)(p_wgt), (const void*)vol.vol_wgt.ptr, sizeof(double )*NP*NP*MP, cudaMemcpyDeviceToHost);
	}
};

class RecRdrWorker : public Worker {
	
public:
	ArgsRec::Info   *p_info;
	float           *p_stack;
	ParticlesSubset *p_ptcls;
	Tomogram        *p_tomo;
	int gpu_ix;
	int max_K;
	int N;
	int M;
	int R;
	int P;
	int pad_type;
	int NP;
	int MP;
	
	double2 **c_acc;
	double  **c_wgt;
	
	float bp_pad;
	
	SubstackCrop    ss_cropper;

	RecRdrWorker() {
		c_acc = NULL;
		c_wgt = NULL;
	}
	
	~RecRdrWorker() {
		if( c_acc != NULL ) {
			for(int r=0;r<R;r++) {
				delete [] c_acc[r];
			}
			delete [] c_acc;
		}
		
		if( c_wgt != NULL ) {
			for(int r=0;r<R;r++) {
				delete [] c_wgt[r];
			}
			delete [] c_wgt;
		}
	}
	
	void setup_global_data(int id,int n_refs,int in_max_K,ArgsRec::Info*info,WorkerCommand*in_worker_cmd) {
		worker_id  = id;
		worker_cmd = in_worker_cmd;
		
		p_info   = info;
		gpu_ix   = info->p_gpu[ id % info->n_threads ];
		max_K    = in_max_K;
		pad_type = info->pad_type;

		
		N = info->box_size;
		M = (N/2)+1;
		R = n_refs;
		P = info->pad_size;
		
		NP = N + P;
		MP = (NP/2)+1;
		
		c_acc = new double2*[R];
		c_wgt = new double *[R];
		for(int r=0;r<R;r++) {
			c_acc[r] = new double2[NP*MP*NP];
			c_wgt[r] = new double [NP*MP*NP];
		}
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
		RecBuffer buffer_a(N,max_K);
		RecBuffer buffer_b(N,max_K);
		PBarrier local_barrier(2);
		DoubleBufferHandler stack_buffer((void*)&buffer_a,(void*)&buffer_b,&local_barrier);
		
		RecGpuWorker gpu_worker;
		init_processing_worker(gpu_worker,&stack_buffer);
		
		int current_cmd;
		
		while( (current_cmd = worker_cmd->read_command()) >= 0 ) {
			switch(current_cmd) {
				case REC_EXEC:
					crop_loop(stack_buffer,stream);
					break;
				default:
					break;
			}
		}
		gpu_worker.wait();
	}
	
	void init_processing_worker(RecGpuWorker&gpu_worker,DoubleBufferHandler*stack_buffer) {
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
		gpu_worker.pad_type   = pad_type;
		gpu_worker.ctf_type   = p_info->ctf_type;
		gpu_worker.max_K      = max_K;
		gpu_worker.c_acc      = c_acc;
		gpu_worker.c_wgt      = c_wgt;
		gpu_worker.bandpass.x = max(bp_scale*p_info->fpix_min-bp_pad,0.0);
		gpu_worker.bandpass.y = min(bp_scale*p_info->fpix_max+bp_pad,(float)NP);
		gpu_worker.bandpass.z = sqrt(p_info->fpix_roll);
		gpu_worker.ssnr.x     = p_info->ssnr_F;
		gpu_worker.ssnr.y     = p_info->ssnr_S;
		gpu_worker.start();
	}
	
	void crop_loop(DoubleBufferHandler&stack_buffer,GPU::Stream&stream) {
		stack_buffer.WO_sync(EMPTY);
		for(int i=worker_id;i<p_ptcls->n_ptcl;i+=p_info->n_threads) {
			RecBuffer*ptr = (RecBuffer*)stack_buffer.WO_get_buffer();
			p_ptcls->get(ptr->ptcl,i);
			if( check_reference(ptr) ) {
				read_defocus(ptr);
				crop_substack(ptr);
				work_progress++;
				work_accumul++;
				if( check_substack(ptr) ) {
					upload(ptr,stream.strm);
					stream.sync();
					stack_buffer.WO_sync(READY);
				}
			}
			else {
				work_progress++;
				work_accumul++;
			}
		}
		stack_buffer.WO_sync(DONE);
	}
	
	void read_defocus(RecBuffer*ptr) {
		ptr->K = p_tomo->stk_dim.z;
		
                float lambda = Math::get_lambda( p_tomo->KV );

		ptr->ctf_vals.AC = p_tomo->AC;
		ptr->ctf_vals.CA = sqrt(1-p_tomo->AC*p_tomo->AC);
		ptr->ctf_vals.apix = p_tomo->pix_size;
		ptr->ctf_vals.LambdaPi = M_PI*lambda;
		ptr->ctf_vals.CsLambda3PiH = lambda*lambda*lambda*(p_tomo->CS*1e7)*M_PI/2;
		
		for(int k=0;k<ptr->K;k++) {
			if( ptr->ptcl.def[k].max_res > 0 ) {
				ptr->ptcl.def[k].max_res = ((float)NP)*p_tomo->pix_size/ptr->ptcl.def[k].max_res;
				ptr->ptcl.def[k].max_res = min(ptr->ptcl.def[k].max_res+bp_pad,(float)NP/2);
			}
		}

		memcpy( (void**)(ptr->c_def.ptr), (const void**)(ptr->ptcl.def), sizeof(Defocus)*ptr->K  );
	}
	
	bool check_reference(RecBuffer*ptr) {
		int r = ptr->ptcl.ref_cix();
		if( p_info->rec_halves )
			ptr->r_ix = 2*r + (ptr->ptcl.half_id()-1);
		else
			ptr->r_ix = r;
		
                if( p_info->rec_halves )
                    return ( ptr->ptcl.ali_w[r] >0 ) && ( ptr->ptcl.half_id() > 0 );
                else
                    return ( ptr->ptcl.ali_w[r] )>0;
	}
	
	void crop_substack(RecBuffer*ptr) {
		V3f pt_tomo,pt_stack,pt_crop,pt_subpix,eu_ZYZ;
		M33f R_tmp,R_ali,R_stack,R_gpu;
		
		int r = ptr->ptcl.ref_cix();
		
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
				//pt_subpix(0) = pt_crop(0) - floor(pt_crop(0));
				//pt_subpix(1) = pt_crop(1) - floor(pt_crop(1));
				//pt_subpix(2) = 0;
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
					if( p_info->norm_type == ArgsRec::NormalizationType_t::NO_NORM ) {
						Math::get_avg_std(ptr->c_pad.ptr[k].x,ptr->c_pad.ptr[k].y,ptr->c_stk.ptr,N*N);
					}
					if( p_info->norm_type == ArgsRec::NormalizationType_t::ZERO_MEAN ) {
						ptr->c_pad.ptr[k].x = 0;
						ptr->c_pad.ptr[k].y = ss_cropper.normalize_zero_mean(ptr->c_stk.ptr,k);
					}
					if( p_info->norm_type == ArgsRec::NormalizationType_t::ZERO_MEAN_1_STD ) {
						ptr->c_pad.ptr[k].x = 0;
						ptr->c_pad.ptr[k].y = 1;
						ss_cropper.normalize_zero_mean_one_std(ptr->c_stk.ptr,k);
					}
					if( p_info->norm_type == ArgsRec::NormalizationType_t::ZERO_MEAN_W_STD ) {
						ptr->c_pad.ptr[k].x = 0;
						ptr->c_pad.ptr[k].y = ptr->ptcl.prj_w[k];
                                                //ss_cropper.normalize_zero_mean_w_std(ptr->c_stk.ptr,ptr->ptcl.prj_w[k],k); /// applying wait at the insertion level
                                                ss_cropper.normalize_zero_mean_one_std(ptr->c_stk.ptr,k);
					}
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
		
	bool check_substack(RecBuffer*ptr) {
		bool rslt = false;
		for(int k=0;k<ptr->K;k++) {
			if( ptr->c_ali.ptr[k].w > 0 )
				rslt = true;
		}
		return rslt;
	}

	void upload(RecBuffer*ptr,cudaStream_t&strm) {
		GPU::upload_async(ptr->g_stk.ptr,ptr->c_stk.ptr,N*N*max_K,strm);
		GPU::upload_async(ptr->g_pad.ptr,ptr->c_pad.ptr,max_K    ,strm);
		GPU::upload_async(ptr->g_ali.ptr,ptr->c_ali.ptr,max_K    ,strm);
		GPU::upload_async(ptr->g_def.ptr,ptr->c_def.ptr,max_K    ,strm);
	}

};

class RecPool : public PoolCoordinator {

public:
	RecRdrWorker  *workers;
	ArgsRec::Info *p_info;
	WorkerCommand w_cmd;
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
	
	RecPool(ArgsRec::Info*info,int n_refs,int in_max_K,int num_ptcls,StackReader&stkrdr,int in_num_threads)
	 : PoolCoordinator(stkrdr,in_num_threads), w_cmd(2*in_num_threads+1) {
		workers  = new RecRdrWorker[in_num_threads];
		p_info   = info;
		max_K    = in_max_K;
		n_ptcls  = num_ptcls;
		N = info->box_size;
		M = (N/2)+1;
		R = n_refs;
		P = info->pad_size;
		NP = N+P;
		MP = (NP/2)+1;
		if( info->rec_halves )
			R = 2*R;
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

	}
	
	~RecPool() {
		delete [] workers;
	}
	
protected:

	void coord_init() {
		for(int i=0;i<p_info->n_threads;i++) {
			workers[i].setup_global_data(i,R,max_K,p_info,&w_cmd);
			workers[i].start();
		}
		progress_start();
	}

	void coord_main(float*stack,ParticlesSubset&ptcls,Tomogram&tomo) {

		w_cmd.presend_sync();
		for(int i=0;i<p_info->n_threads;i++) {
			workers[i].setup_working_data(stack,&ptcls,&tomo);
		}
		w_cmd.send_command(RecCmd::REC_EXEC);
		
		show_progress(ptcls.n_ptcl);
		
		w_cmd.send_command(WorkerCommand::BasicCommands::CMD_IDLE);
		
	}
	
	void coord_end() {
		show_done();
		w_cmd.send_command(WorkerCommand::BasicCommands::CMD_END);
		for(int i=0;i<p_info->n_threads;i++) {
			workers[i].wait();
		}
		
		gather_results();
		reconstruct_results();
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
		sprintf(progress_buffer,"        Filling fourier space: Buffering...");
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
				sprintf(progress_buffer,"        Filling fourier space: %6.2f%%%%",progress_percent);
				int n = strlen(progress_buffer);
                                add_etc(progress_buffer+n,progress,n_ptcls);
			}
			else {
				sprintf(progress_buffer,"        Filling fourier space: Buffering...");
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
		sprintf(progress_buffer,"        Filling fourier space: 100.00%%%%");
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

	void gather_results() {
		int l = NP*NP*MP;
		for(int i=1;i<p_info->n_threads;i++) {
			for(int r=0;r<R;r++) {
				Math::sum(workers[0].c_acc[r],workers[i].c_acc[r],l);
				Math::sum(workers[0].c_wgt[r],workers[i].c_wgt[r],l);
			}
                }
	}
	
        virtual void reconstruct_results() {
		GPU::set_device( p_info->p_gpu[0] );
		
		GPU::GArrDouble  p_wgt;
		GPU::GArrDouble2 p_acc;
		GPU::GArrSingle  p_vol;
		float*vol;
		
		p_wgt.alloc(MP*NP*NP);
		p_acc.alloc(MP*NP*NP);
		p_vol.alloc(N*N*N);
		vol = new float[N*N*N];
		
		if( p_info->rec_halves )
			reconstruct_halves(vol,p_vol,p_acc,p_wgt);
		else
			reconstruct_maps(vol,p_vol,p_acc,p_wgt);
			
		delete [] vol;
	}
	
	void reconstruct_maps(float*vol,GPU::GArrSingle&p_vol,GPU::GArrDouble2&p_acc,GPU::GArrDouble&p_wgt) {
                char out_file[SUSAN_FILENAME_LENGTH];
		for(int r=0;r<R;r++) {
			sprintf(out_file,"%s_class%03d.mrc",p_info->out_pfx,r+1);
			printf("        Reconstructing %s ... ",out_file); fflush(stdout);
                        reconstruct_upload(workers[0].c_acc[r],workers[0].c_wgt[r],p_acc,p_wgt);
                        reconstruct_sym(p_acc,p_wgt);
                        reconstruct_invert(p_wgt);
                        reconstruct_core(p_vol,p_acc,p_wgt);
			reconstruct_download(vol,p_vol);
                        Mrc::write(vol,N,N,N,out_file);
                        Mrc::set_apix(out_file,tomos->at(0).pix_size,N,N,N);
			printf(" Done.\n");
		}
	}
	
	void reconstruct_halves(float*vol,GPU::GArrSingle&p_vol,GPU::GArrDouble2&p_acc,GPU::GArrDouble&p_wgt) {
		int l = NP*NP*MP;
                char out_file[SUSAN_FILENAME_LENGTH];
		for(int r=0;r<R/2;r++) {
			
			sprintf(out_file,"%s_class%03d_half1.mrc",p_info->out_pfx,r+1);
			printf("        Reconstructing %s ... ",out_file); fflush(stdout);
			reconstruct_upload(workers[0].c_acc[2*r  ],workers[0].c_wgt[2*r  ],p_acc,p_wgt);
                        reconstruct_sym(p_acc,p_wgt);
                        reconstruct_invert(p_wgt);
                        reconstruct_core(p_vol,p_acc,p_wgt);
			reconstruct_download(vol,p_vol);
			Mrc::write(vol,N,N,N,out_file);
                        Mrc::set_apix(out_file,tomos->at(0).pix_size,N,N,N);
			printf(" Done.\n");
			
			sprintf(out_file,"%s_class%03d_half2.mrc",p_info->out_pfx,r+1);
			printf("        Reconstructing %s ... ",out_file); fflush(stdout);
			reconstruct_upload(workers[0].c_acc[2*r+1],workers[0].c_wgt[2*r+1],p_acc,p_wgt);
                        reconstruct_sym(p_acc,p_wgt);
                        reconstruct_invert(p_wgt);
                        reconstruct_core(p_vol,p_acc,p_wgt);
			reconstruct_download(vol,p_vol);
			Mrc::write(vol,N,N,N,out_file);
                        Mrc::set_apix(out_file,tomos->at(0).pix_size,N,N,N);
			printf(" Done.\n");
			
			Math::sum(workers[0].c_acc[2*r],workers[0].c_acc[2*r+1],l);
			Math::sum(workers[0].c_wgt[2*r],workers[0].c_wgt[2*r+1],l);
			sprintf(out_file,"%s_class%03d.mrc",p_info->out_pfx,r+1);
			printf("        Reconstructing %s ... ",out_file); fflush(stdout);
			reconstruct_upload(workers[0].c_acc[2*r  ],workers[0].c_wgt[2*r  ],p_acc,p_wgt);
                        reconstruct_sym(p_acc,p_wgt);
                        reconstruct_invert(p_wgt);
                        reconstruct_core(p_vol,p_acc,p_wgt);
			reconstruct_download(vol,p_vol);
			Mrc::write(vol,N,N,N,out_file);
                        Mrc::set_apix(out_file,tomos->at(0).pix_size,N,N,N);
			printf(" Done.\n");
			
		}
	}
	
	void reconstruct_upload(double2*c_acc,double*c_wgt,GPU::GArrDouble2&p_acc,GPU::GArrDouble&p_wgt) {
		cudaMemcpy((void*)p_acc.ptr,(const void*)c_acc,sizeof(double2)*MP*NP*NP,cudaMemcpyHostToDevice);
		cudaMemcpy((void*)p_wgt.ptr,(const void*)c_wgt,sizeof(double )*MP*NP*NP,cudaMemcpyHostToDevice);
	}

        void reconstruct_sym(GPU::GArrDouble2&p_acc,GPU::GArrDouble&p_wgt) {
            RecSym sym(MP,NP,p_info->sym);
            sym.apply_sym(p_acc,p_wgt);
        }

        void reconstruct_invert(GPU::GArrDouble&p_wgt) {
            RecInvWgt inv_wgt(NP,MP,p_info->w_inv_ite,p_info->w_inv_std);
            inv_wgt.invert(p_wgt);
        }
	
        void reconstruct_core(GPU::GArrSingle&p_vol,GPU::GArrDouble2&p_acc,GPU::GArrDouble&p_wgt) {
            RecInvVol inv_vol(N,P);
            inv_vol.apply_inv_wgt(p_acc,p_wgt);
            inv_vol.invert_and_extract(p_vol);
	}
	
	void reconstruct_download(float*vol,GPU::GArrSingle&p_vol) {
		cudaMemcpy((void*)vol,(const void*)p_vol.ptr,sizeof(float)*N*N*N,cudaMemcpyDeviceToHost);
		float avg,std;
		Math::get_avg_std(avg,std,vol,N*N*N);
                if( !Math::normalize(vol,N*N*N,avg,std,1.0) ) {
			Math::randn(vol,N*N*N);
			printf("(Empty, filling with noise)");
		}
			
	}

};

#endif /// RECONSTRUCT_H


