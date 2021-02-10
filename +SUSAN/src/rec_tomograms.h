#ifndef REC_TOMOGRAMS_H
#define REC_TOMOGRAMS_H

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
#include "em.h"
#include "io.h"
#include "points_provider.h"
#include "rec_acc.h"
#include "rec_tomograms_args.h"
#include "tomo_generator.h"

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
	
	RecBuffer(int N,int max_k) {
		c_stk.alloc(N*N*max_k);
		g_stk.alloc(N*N*max_k);
		c_pad.alloc(max_k);
		g_pad.alloc(max_k);
		c_ali.alloc(max_k);
		g_ali.alloc(max_k);
		c_def.alloc(max_k);
		g_def.alloc(max_k);
	}
	
	~RecBuffer() {
	}
	
};

class RecTomogramsWorker : public Worker {
	
public:
	ArgsRecTomos::Info *p_info;
	float              *p_stack;
	ParticlesSubset    *p_ptcls;
	Tomogram           *p_tomo;
	int gpu_ix;
	int max_K;
	int N;
	int M;
	int P;
	int pad_type;
	int ctf_type;
	float3  bandpass;
	float2  ssnr; /// x=F; y=S;
	
	int NP;
	int MP;
	
	int    w_inv_ite;
    float  w_inv_std;
    
	float bp_pad;
	
	int maxXY;
	
	SubstackCrop ss_cropper;

	RecTomogramsWorker() {
	}
	
	~RecTomogramsWorker() {
	}
	
	void setup_global_data(int id,int in_max_K,int in_maxXY,ArgsRecTomos::Info*info,WorkerCommand*in_worker_cmd) {
		worker_id  = id;
		worker_cmd = in_worker_cmd;
		
		p_info   = info;
		gpu_ix   = info->p_gpu[ id % info->n_threads ];
		max_K    = in_max_K;
		pad_type = info->pad_type;
		maxXY    = in_maxXY;

		
		N = info->box_size;
		M = (N/2)+1;
		P = info->pad_size;
		
		NP = N + P;
                MP = (NP/2)+1;
		
		ctf_type = info->ctf_type;
		
		bp_pad = info->fpix_roll/2;
                float bp_scale = ((float)N)/((float)NP);
		bandpass.x = max(bp_scale*info->fpix_min-bp_pad,0.0);
		bandpass.y = min(bp_scale*info->fpix_max+bp_pad,(float)NP);
		bandpass.z = sqrt(info->fpix_roll);
		ssnr.x     = info->ssnr_F;
		ssnr.y     = info->ssnr_S;
		w_inv_ite  = info->w_inv_ite;
                w_inv_std  = info->w_inv_std;
	}
	
	void setup_working_data(float*stack,ParticlesSubset*ptcls,Tomogram*tomo) {
		p_stack = stack;
		p_ptcls = ptcls;
		p_tomo  = tomo;
                ss_cropper.setup(tomo,NP);
		work_progress=0;
	}
	
protected:
	void main() {
		
		GPU::set_device(gpu_ix);
		GPU::Stream stream;
		stream.configure();
		work_accumul = 0;
                RecBuffer buffer(NP,max_K);
		
                RecSubstack ss_data(MP,NP,max_K,0,stream);
		RecInvWgt inv_wgt(NP,MP,w_inv_ite,w_inv_std);
		RecInvVol inv_vol(N,P);
		RecAcc vol;
		vol.alloc(MP,NP,max_K);
		
		GPU::GArrSingle p_vol;
		p_vol.alloc(N*N*N);
		
		float*map = new float[N*N*N];
		
		int current_cmd;
		
		TomoRec tomo_rec(N,maxXY);
		
		while( (current_cmd = worker_cmd->read_command()) >= 0 ) {
			switch(current_cmd) {
				case REC_EXEC:
					crop_loop(tomo_rec,map,p_vol,inv_wgt,inv_vol,ss_data,vol,buffer,stream);
					break;
				default:
					break;
			}
		}
		
		delete [] map;
	}
	
	void crop_loop(TomoRec&tomo_rec,float*map,GPU::GArrSingle&p_vol,RecInvWgt&inv_wgt,RecInvVol&inv_vol,RecSubstack&ss_data,RecAcc&vol,RecBuffer&buffer,GPU::Stream&stream) {
		V3f pt_tomo;
		char filename[SUSAN_FILENAME_LENGTH];
		sprintf(filename,"%s_%03d.mrc",p_info->out_pfx,p_tomo->tomo_id);
		tomo_rec.start_rec(filename,p_tomo);
                for(int i=worker_id;i<p_ptcls->n_ptcl;i+=p_info->n_threads) {
			p_ptcls->get(buffer.ptcl,i);
			read_defocus(buffer);
			crop_substack(pt_tomo,buffer);
			work_progress++;
			if( check_substack(buffer) ) {
				vol.clear();
				upload(buffer,stream.strm);
				add_data(ss_data,buffer,stream);
				correct_ctf(ss_data,buffer,stream);
				insert_vol(vol,ss_data,buffer,stream);
				stream.sync();
				reconstruct_core(p_vol,inv_wgt,inv_vol,vol.vol_acc,vol.vol_wgt);
				cudaMemcpy((void*)map,(const void*)p_vol.ptr,sizeof(float)*N*N*N,cudaMemcpyDeviceToHost);
				tomo_rec.add_block(map,pt_tomo);
			}
                }
		tomo_rec.end_rec();
	}
	
	void read_defocus(RecBuffer&ptr) {
		ptr.K = p_tomo->stk_dim.z;
		
		float lambda = Math::get_lambda( p_tomo->KV );
		
		ptr.ctf_vals.AC = p_tomo->AC;
		ptr.ctf_vals.CA = sqrt(1-p_tomo->AC*p_tomo->AC);
		ptr.ctf_vals.apix = p_tomo->pix_size;
		ptr.ctf_vals.LambdaPi = M_PI*lambda;
		ptr.ctf_vals.CsLambda3PiH = lambda*lambda*lambda*(p_tomo->CS*1e7)*M_PI/2;
	}
	
	void crop_substack(V3f&pt_work,RecBuffer&ptr) {
		V3f pt_tomo,pt_stack,pt_crop,pt_subpix,eu_ZYZ;
		M33f R_tmp,R_stack,R_gpu;
		
                /// P_tomo = P_ptcl
		pt_tomo(0) = ptr.ptcl.pos().x;
		pt_tomo(1) = ptr.ptcl.pos().y;
		pt_tomo(2) = ptr.ptcl.pos().z;
		
		pt_work(0) = round(pt_tomo(0)/p_tomo->pix_size + p_tomo->tomo_center(0));
		pt_work(1) = round(pt_tomo(1)/p_tomo->pix_size + p_tomo->tomo_center(1));
		pt_work(2) = round(pt_tomo(2)/p_tomo->pix_size + p_tomo->tomo_center(2));
		
                for(int k=0;k<ptr.K;k++) {
			if( ptr.ptcl.prj_w[k] > 0 ) {
				
				/// P_stack = R^k_tomo*P_tomo + t^k_tomo
                                pt_stack = p_tomo->R[k]*pt_tomo + p_tomo->t[k];
                                Math::Rmat_eZYZ(eu_ZYZ,p_tomo->R[k]);

				/// P_crop = R^k_prj*P_stack + t^k_prj
				eu_ZYZ(0) = ptr.ptcl.prj_eu[k].x;
				eu_ZYZ(1) = ptr.ptcl.prj_eu[k].y;
				eu_ZYZ(2) = ptr.ptcl.prj_eu[k].z;
                                Math::eZYZ_Rmat(R_tmp,eu_ZYZ);
				pt_crop = R_tmp*pt_stack;
				pt_crop(0) += ptr.ptcl.prj_t[k].x;
				pt_crop(1) += ptr.ptcl.prj_t[k].y;
				
				/// R_stack = R^k_prj*R^k_tomo
				R_stack = R_tmp*p_tomo->R[k];
				
				/// Set defocus
                                ptr.c_def.ptr[k].U = p_tomo->def[k].U - pt_crop(2);
                                ptr.c_def.ptr[k].V = p_tomo->def[k].V - pt_crop(2);
				ptr.c_def.ptr[k].angle = p_tomo->def[k].angle;
				
				/// Angstroms -> pixels
                                pt_crop = pt_crop/p_tomo->pix_size + p_tomo->stk_center;
                                pt_crop(0) += 0.5;
                                pt_crop(1) += 0.5;
                                pt_crop(2) += 0.5;

                                /// Get subpixel shift
				V3f pt_tmp;
				pt_tmp(0) = pt_crop(0) - floor(pt_crop(0));
				pt_tmp(1) = pt_crop(1) - floor(pt_crop(1));
				pt_tmp(2) = 0;
				pt_subpix = R_stack.transpose()*pt_tmp;
				
				/// Setup data for upload to GPU
				ptr.c_ali.ptr[k].t.x = -pt_subpix(0);
				ptr.c_ali.ptr[k].t.y = -pt_subpix(1);
				ptr.c_ali.ptr[k].t.z = 0;
				ptr.c_ali.ptr[k].w = ptr.ptcl.prj_w[k];
                                R_gpu = R_stack.transpose();
                                Math::set( ptr.c_ali.ptr[k].R, R_gpu );

				/// Crop
				if( ss_cropper.check_point(pt_crop) ) {
					ss_cropper.crop(ptr.c_stk.ptr,p_stack,pt_crop,k);
					if( p_info->norm_type == ArgsRec::NormalizationType_t::NO_NORM ) {
						Math::get_avg_std(ptr.c_pad.ptr[k].x,ptr.c_pad.ptr[k].y,ptr.c_stk.ptr,N*N);
					}
					if( p_info->norm_type == ArgsRec::NormalizationType_t::ZERO_MEAN ) {
						ptr.c_pad.ptr[k].x = 0;
						ptr.c_pad.ptr[k].y = ss_cropper.normalize_zero_mean(ptr.c_stk.ptr,k);
					}
					if( p_info->norm_type == ArgsRec::NormalizationType_t::ZERO_MEAN_1_STD ) {
						ptr.c_pad.ptr[k].x = 0;
						ptr.c_pad.ptr[k].y = 1;
						ss_cropper.normalize_zero_mean_one_std(ptr.c_stk.ptr,k);
					}
					if( p_info->norm_type == ArgsRec::NormalizationType_t::ZERO_MEAN_W_STD ) {
						ptr.c_pad.ptr[k].x = 0;
						ptr.c_pad.ptr[k].y = ptr.ptcl.prj_w[k];
						ss_cropper.normalize_zero_mean_w_std(ptr.c_stk.ptr,ptr.ptcl.prj_w[k],k);
					}
					ptr.ptcl.prj_w[k] = ptr.c_pad.ptr[k].y;
					
				}
				else {
					ptr.c_ali.ptr[k].w = 0;
				}
			}
			else {
				ptr.c_ali.ptr[k].w = 0;
			}
		}
	}
		
	bool check_substack(RecBuffer&ptr) {
		bool rslt = false;
		for(int k=0;k<ptr.K;k++) {
			if( ptr.c_ali.ptr[k].w > 0 )
				rslt = true;
		}
		return rslt;
	}

        void upload(RecBuffer&ptr,cudaStream_t&strm) {
                GPU::upload_async(ptr.g_stk.ptr,ptr.c_stk.ptr,NP*NP*max_K,strm);
		GPU::upload_async(ptr.g_pad.ptr,ptr.c_pad.ptr,max_K    ,strm);
		GPU::upload_async(ptr.g_ali.ptr,ptr.c_ali.ptr,max_K    ,strm);
		GPU::upload_async(ptr.g_def.ptr,ptr.c_def.ptr,max_K    ,strm);
	}

	void add_data(RecSubstack&ss_data,RecBuffer&ptr,GPU::Stream&stream) {
		if( pad_type == ArgsRec::PaddingType_t::PAD_ZERO )
			ss_data.pad_zero(stream);
		if( pad_type == ArgsRec::PaddingType_t::PAD_GAUSSIAN )
			ss_data.pad_normal(ptr.g_pad,ptr.K,stream);
		
		ss_data.add_data(ptr.g_stk,ptr.g_ali,ptr.K,stream);
	}
	
	void correct_ctf(RecSubstack&ss_data,RecBuffer&ptr,GPU::Stream&stream) {
		if( ctf_type == ArgsRec::InversionType_t::NO_INV )
			ss_data.set_no_ctf(bandpass,ptr.K,stream);
		if( ctf_type == ArgsRec::InversionType_t::PHASE_FLIP )
			ss_data.set_phase_flip(ptr.ctf_vals,ptr.g_def,bandpass,ptr.K,stream);
		if( ctf_type == ArgsRec::InversionType_t::WIENER_INV )
			ss_data.set_wiener(ptr.ctf_vals,ptr.g_def,bandpass,ptr.K,stream);
		if( ctf_type == ArgsRec::InversionType_t::WIENER_INV_SSNR )
			ss_data.set_wiener_ssnr(ptr.ctf_vals,ptr.g_def,bandpass,ssnr,ptr.K,stream);
			
	}

	void insert_vol(RecAcc&vol,RecSubstack&ss_data,RecBuffer&ptr,GPU::Stream&stream) {
                vol.insert(ss_data.ss_tex,ss_data.ss_ctf,ptr.g_ali,bandpass,ptr.K,stream);
	}
	
	void reconstruct_core(GPU::GArrSingle&p_vol,RecInvWgt&inv_wgt,RecInvVol&inv_vol,GPU::GArrDouble2&p_acc,GPU::GArrDouble&p_wgt) {
		inv_wgt.invert(p_wgt);
		inv_vol.apply_inv_wgt(p_acc,p_wgt);
		inv_vol.invert_and_extract(p_vol);
	}
};

class RecTomogramsPool : public PoolCoordinator {

public:
	RecTomogramsWorker  *workers;
	ArgsRecTomos::Info *p_info;
	WorkerCommand w_cmd;
	int max_K;
	int N;
	int M;
	int P;
	int NP;
	int MP;
	
	int maxXY;
	
	RecTomogramsPool(ArgsRecTomos::Info*info,int in_max_K,StackReader&stkrdr,int in_num_threads)
	 : PoolCoordinator(stkrdr,in_num_threads), w_cmd(in_num_threads+1) {
		workers  = new RecTomogramsWorker[in_num_threads];
		p_info   = info;
		max_K    = in_max_K;
		N = info->box_size;
		M = (N/2)+1;
		P = info->pad_size;
		NP = N+P;
		MP = (NP/2)+1;
		
		maxXY = 0;
		for(int t=0;t<stkrdr.tomos->num_tomo;t++) {
			int cur_numel = stkrdr.tomos->at(t).tomo_dim.x*stkrdr.tomos->at(t).tomo_dim.y;
			if( maxXY < cur_numel )
				maxXY = cur_numel;
		}
		
	}
	
	~RecTomogramsPool() {
		delete [] workers;
	}
	
protected:

	void coord_init() {
		for(int i=0;i<p_info->n_threads;i++) {
			workers[i].setup_global_data(i,max_K,maxXY,p_info,&w_cmd);
			workers[i].start();
		}
	}

	void coord_main(float*stack,ParticlesSubset&ptcls,Tomogram&tomo) {

		int count=0;

		printf("        Reconstructing tomo %d: %6.2f%%",tomo.tomo_id,0);
		fflush(stdout);

		w_cmd.presend_sync();
		for(int i=0;i<p_info->n_threads;i++) {
			workers[i].setup_working_data(stack,&ptcls,&tomo);
		}
		w_cmd.send_command(RecCmd::REC_EXEC);
		
                while( (count=count_progress()) < ptcls.n_ptcl ) {
			printf("\b\b\b\b\b\b\b%6.2f%%",100*float(count)/float(ptcls.n_ptcl));
			fflush(stdout);
			sleep(1);
		}
                printf("\b\b\b\b\b\b\b100.00%%\n"); fflush(stdout);
		
		w_cmd.presend_sync();
		clear_workers();
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
	
	void clear_workers() {
		for(int i=1;i<p_info->n_threads;i++)
			workers[i].work_progress = 0;
	}
};

#endif /// REC_TOMOGRAMS_H


