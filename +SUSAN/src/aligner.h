#ifndef ALIGNER_H
#define ALIGNER_H

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
#include "aligner_args.h"

typedef enum {
	ALI_3D=1,
	ALI_2D
} AliCmd;

class AliBuffer {
	
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
	
	AliBuffer(int N,int max_k) {
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
	
	~AliBuffer() {
	}
	
};

class AliGpuWorker : public Worker {
	
public:
	int gpu_ix;
	int N;
	int M;
	int P;
	int R;
	int pad_type;
	int ctf_type;
	int max_K;
	bool ali_halves;
	float3  bandpass;
	float2  ssnr; /// x=F; y=S;
	DoubleBufferHandler *p_buffer;
        RefMap              *p_refs;
	
        const char*psym;
        float2 cone; /// x=range; y=step
        float2 inplane; /// x=range; y=step
        uint32 ref_level;
        uint32 ref_factor;
        uint32 off_type;
        float4 off_par;

	bool drift2D;
	bool drift3D;

        AnglesProvider ang_prov;
	
	AliGpuWorker() {
	}
	
	~AliGpuWorker() {
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

                AliData ali_data(MP,NP,max_K,off_par,off_type);

                GPU::GArrSingle ctf_wgt;
                ctf_wgt.alloc(MP*NP*max_K);

                int num_vols = R;
                if( ali_halves ) num_vols = 2*R;
                AliRef vols[num_vols];
                allocate_references(vols);

                ang_prov.cone_range = cone.x;
                ang_prov.cone_step  = cone.y;
                ang_prov.inplane_range = inplane.x;
                ang_prov.inplane_step  = inplane.y;
                ang_prov.refine_factor = ref_factor;
                ang_prov.refine_level  = ref_level;
                ang_prov.set_symmetry(psym);

                GPU::sync();

		while( (current_cmd = worker_cmd->read_command()) >= 0 ) {
			switch(current_cmd) {
				case ALI_3D:
                                        align3D(vols,ctf_wgt,ss_data,ali_data,stream);
					break;
				default:
					break;
			}
		}
		
		GPU::sync();
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

            if( ali_halves ) {
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

        void align3D(AliRef*vols,GPU::GArrSingle&ctf_wgt,AliSubstack&ss_data,AliData&ali_data,GPU::Stream&stream) {
		p_buffer->RO_sync();
		while( p_buffer->RO_get_status() > DONE ) {
			if( p_buffer->RO_get_status() == READY ) {
				AliBuffer*ptr = (AliBuffer*)p_buffer->RO_get_buffer();
                                create_ctf(ctf_wgt,ptr,stream);
                                add_data(ss_data,ctf_wgt,ptr,stream);
                                angular_search_3D(vols[ptr->r_ix],ss_data,ctf_wgt,ptr,ali_data,stream);
                                //correct_ctf(ss_data,ptr,stream);
				//insert_vol(vols[ptr->r_ix],ss_data,ptr,stream);
                                stream.sync();
			}
			p_buffer->RO_sync();
		}
	}

        void create_ctf(GPU::GArrSingle&ctf_wgt,AliBuffer*ptr,GPU::Stream&stream) {
            int3 ss = make_int3(MP,NP,ptr->K);
            dim3 blk = GPU::get_block_size_2D();
            dim3 grd = GPU::calc_grid_size(blk,MP,NP,ptr->K);
            GpuKernelsCtf::create_ctf<<<grd,blk,0,stream.strm>>>(ctf_wgt.ptr,ptr->ctf_vals,ptr->g_def.ptr,ss);
        }
	
        void add_data(AliSubstack&ss_data,GPU::GArrSingle&ctf_wgt,AliBuffer*ptr,GPU::Stream&stream) {
                if( pad_type == ArgsAli::PaddingType_t::PAD_ZERO )
			ss_data.pad_zero(stream);
                if( pad_type == ArgsAli::PaddingType_t::PAD_GAUSSIAN )
			ss_data.pad_normal(ptr->g_pad,ptr->K,stream);
		
		ss_data.add_data(ptr->g_stk,ptr->g_ali,ptr->K,stream);

                if( ctf_type == ArgsAli::CtfCorrectionType_t::ON_SUBSTACK )
                    ss_data.correct_wiener(ptr->ctf_vals,ctf_wgt,ptr->g_def,bandpass,ptr->K,stream);
                if( ctf_type == ArgsAli::CtfCorrectionType_t::ON_SUBSTACK_SSNR )
                    ss_data.correct_wiener_ssnr(ptr->ctf_vals,ctf_wgt,ptr->g_def,bandpass,ssnr,ptr->K,stream);
                if( ctf_type == ArgsAli::CtfCorrectionType_t::ON_SUBSTACK_WHITENING )
                    ss_data.correct_wiener_whitening(ptr->ctf_vals,ctf_wgt,ptr->g_def,bandpass,ptr->K,stream);

                float w_total=0;
                for(int k=0;k<ptr->K;k++) {
                    w_total += ptr->c_ali.ptr[k].w;
                }

                ss_data.apply_radial_wgt(w_total,ptr->K,stream);

	}
	
        void angular_search_3D(AliRef&vol,AliSubstack&ss_data,GPU::GArrSingle&ctf_wgt,AliBuffer*ptr,AliData&ali_data,GPU::Stream&stream) {
            Rot33 R;
            //single max_cc=0,cc;
            //int max_idx=0,idx;
            M33f max_R,R_ite,R_tmp;
            M33f R_lvl = Eigen::MatrixXf::Identity(3,3);

            for( ang_prov.levels_init(); ang_prov.levels_available(); ang_prov.levels_next() ) {
                for( ang_prov.sym_init(); ang_prov.sym_available(); ang_prov.sym_next() ) {
                    for( ang_prov.cone_init(); ang_prov.cone_available(); ang_prov.cone_next() ) {
                        for( ang_prov.inplane_init(); ang_prov.inplane_available(); ang_prov.inplane_next() ) {

                            ang_prov.get_current_R(R_ite);
                            R_tmp = R_ite*R_lvl;
                            Math::set(R,R_tmp);
                            ali_data.rotate_post(R,ptr->g_ali,ptr->K,stream);
                            ali_data.project(vol.ref,bandpass,ptr->K,stream);

                            /*g_refs[vol_ix].extract_stack(g_ext.g_tlt,g_ext.g_stk_f,gpu_in.cur_K,stream_process);
                            g_ext.cross_correlate_and_fft(g_stk.g_stk_f,g_ctf.g_ptr,gpu_in.cur_K,stream_process);
                            g_cc.fftshift_and_set(g_ext.g_stk_r,gpu_in.cur_K,stream_process);
                            g_wbp.reconstruct(g_cc.text_obj,g_ext.g_tlt,gpu_in.cur_K,stream_process);
                            g_wbp.get_max(cc,idx);
                            if( cc > max_cc ) {
                                max_idx = idx;
                                max_cc  = cc;
                                max_R   = R_tmp;
                            }*/
                        }
                    }
                }
                R_lvl = max_R;
            }
        }

        /*void insert_loop(RecAcc*vols,RecSubstack&ss_data,GPU::Stream&stream) {
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
	}*/
};

class AliRdrWorker : public Worker {
	
public:
	ArgsAli::Info   *p_info;
	float           *p_stack;
	ParticlesSubset *p_ptcls;
	Tomogram        *p_tomo;
        RefMap          *p_refs;
	int gpu_ix;
	int max_K;
	int N;
	int M;
	int R;
	int P;
	int pad_type;
	int NP;
	int MP;
	
	bool drift2D;
	bool drift3D;
	
	float bp_pad;
	
	SubstackCrop    ss_cropper;

	AliRdrWorker() {
	}
	
	~AliRdrWorker() {
	}
	
        void setup_global_data(int id,RefMap*in_p_refs,int in_R,int in_max_K,ArgsAli::Info*info,WorkerCommand*in_worker_cmd) {
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
		
		drift2D = true;
		drift3D = true;
		
		if( info->type == 2 && !info->drift ) drift2D = false;
		if( info->type == 3 && !info->drift ) drift3D = false;
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
		AliBuffer buffer_a(N,max_K);
		AliBuffer buffer_b(N,max_K);
		PBarrier local_barrier(2);
		DoubleBufferHandler stack_buffer((void*)&buffer_a,(void*)&buffer_b,&local_barrier);
		
		AliGpuWorker gpu_worker;
		init_processing_worker(gpu_worker,&stack_buffer);
		
		int current_cmd;
		
		while( (current_cmd = worker_cmd->read_command()) >= 0 ) {
			switch(current_cmd) {
				case ALI_3D:
					crop_loop(stack_buffer,stream);
					break;
				case ALI_2D:
					crop_loop(stack_buffer,stream);
					break;
				default:
					break;
			}
		}
		gpu_worker.wait();
	}
	
	void init_processing_worker(AliGpuWorker&gpu_worker,DoubleBufferHandler*stack_buffer) {
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
		gpu_worker.ali_halves = p_info->ali_halves;
		gpu_worker.pad_type   = pad_type;
		gpu_worker.ctf_type   = p_info->ctf_type;
		gpu_worker.max_K      = max_K;
		gpu_worker.bandpass.x = max(bp_scale*p_info->fpix_min-bp_pad,0.0);
		gpu_worker.bandpass.y = min(bp_scale*p_info->fpix_max+bp_pad,(float)NP);
		gpu_worker.bandpass.z = sqrt(p_info->fpix_roll);
		gpu_worker.ssnr.x     = p_info->ssnr_F;
		gpu_worker.ssnr.y     = p_info->ssnr_S;
		gpu_worker.drift2D    = drift2D;
		gpu_worker.drift3D    = drift3D;
                gpu_worker.cone.x     = p_info->cone_range;
                gpu_worker.cone.y     = p_info->cone_step;
                gpu_worker.inplane.x  = p_info->inplane_range;
                gpu_worker.inplane.y  = p_info->inplane_step;
                gpu_worker.ref_factor = p_info->refine_factor;
                gpu_worker.ref_level  = p_info->refine_level;
                gpu_worker.off_type   = p_info->off_type;
                gpu_worker.off_par.x  = p_info->off_x;
                gpu_worker.off_par.y  = p_info->off_y;
                gpu_worker.off_par.z  = p_info->off_z;
                gpu_worker.off_par.w  = p_info->off_s;
                gpu_worker.psym       = p_info->pseudo_sym;
                gpu_worker.start();
	}
	
	void crop_loop(DoubleBufferHandler&stack_buffer,GPU::Stream&stream) {
		stack_buffer.WO_sync(EMPTY);
		for(int i=worker_id;i<p_ptcls->n_ptcl;i+=p_info->n_threads) {
			AliBuffer*ptr = (AliBuffer*)stack_buffer.WO_get_buffer();
			p_ptcls->get(ptr->ptcl,i);
			read_defocus(ptr);
			work_progress++;
			work_accumul++;
                        for(int r=0;r<R;r++) {
				crop_substack(ptr,r);
				if( check_substack(ptr) ) {
					upload(ptr,stream.strm);
					stream.sync();
					stack_buffer.WO_sync(READY);
				}
			}
		}
		stack_buffer.WO_sync(DONE);
	}
	
	void read_defocus(AliBuffer*ptr) {
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
	
	void crop_substack(AliBuffer*ptr,const int r) {
		V3f pt_tomo,pt_stack,pt_crop,pt_subpix,eu_ZYZ;
		M33f R_tmp,R_ali,R_stack,R_gpu;
		
                if( p_info->ali_halves )
                        ptr->r_ix = 2*r + (ptr->ptcl.half_id()-1);
                else
                        ptr->r_ix = r;
		
		/// P_tomo = P_ptcl + t_ali
		if( drift3D ) {
			pt_tomo(0) = ptr->ptcl.pos().x + ptr->ptcl.ali_t[r].x;
			pt_tomo(1) = ptr->ptcl.pos().y + ptr->ptcl.ali_t[r].y;
			pt_tomo(2) = ptr->ptcl.pos().z + ptr->ptcl.ali_t[r].z;
		}
		else {
			pt_tomo(0) = ptr->ptcl.pos().x;
			pt_tomo(1) = ptr->ptcl.pos().y;
			pt_tomo(2) = ptr->ptcl.pos().z;
		}
		
		
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
				pt_crop = R_tmp*pt_stack;
				if( drift2D ) {
					pt_crop(0) += ptr->ptcl.prj_t[k].x;
					pt_crop(1) += ptr->ptcl.prj_t[k].y;
				}
				
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
					if( p_info->norm_type == ArgsAli::NormalizationType_t::NO_NORM ) {
						Math::get_avg_std(ptr->c_pad.ptr[k].x,ptr->c_pad.ptr[k].y,ptr->c_stk.ptr,N*N);
					}
					if( p_info->norm_type == ArgsAli::NormalizationType_t::ZERO_MEAN ) {
						ptr->c_pad.ptr[k].x = 0;
						ptr->c_pad.ptr[k].y = ss_cropper.normalize_zero_mean(ptr->c_stk.ptr,k);
					}
					if( p_info->norm_type == ArgsAli::NormalizationType_t::ZERO_MEAN_1_STD ) {
						ptr->c_pad.ptr[k].x = 0;
						ptr->c_pad.ptr[k].y = 1;
						ss_cropper.normalize_zero_mean_one_std(ptr->c_stk.ptr,k);
					}
					if( p_info->norm_type == ArgsAli::NormalizationType_t::ZERO_MEAN_W_STD ) {
						ptr->c_pad.ptr[k].x = 0;
						ptr->c_pad.ptr[k].y = ptr->ptcl.prj_w[k];
						ss_cropper.normalize_zero_mean_w_std(ptr->c_stk.ptr,ptr->ptcl.prj_w[k],k);
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

	bool check_substack(AliBuffer*ptr) {
		bool rslt = false;
		for(int k=0;k<ptr->K;k++) {
			if( ptr->c_ali.ptr[k].w > 0 )
				rslt = true;
		}
		return rslt;
	}

	void upload(AliBuffer*ptr,cudaStream_t&strm) {
		GPU::upload_async(ptr->g_stk.ptr,ptr->c_stk.ptr,N*N*max_K,strm);
		GPU::upload_async(ptr->g_pad.ptr,ptr->c_pad.ptr,max_K    ,strm);
		GPU::upload_async(ptr->g_ali.ptr,ptr->c_ali.ptr,max_K    ,strm);
		GPU::upload_async(ptr->g_def.ptr,ptr->c_def.ptr,max_K    ,strm);
	}

};

class AliPool : public PoolCoordinator {

public:
	AliRdrWorker  *workers;
	ArgsAli::Info *p_info;
	WorkerCommand w_cmd;
        RefMap        *p_refs;
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
	
	AliPool(ArgsAli::Info*info,References*in_p_refs,int in_max_K,int num_ptcls,StackReader&stkrdr,int in_num_threads)
	 : PoolCoordinator(stkrdr,in_num_threads), w_cmd(2*in_num_threads+1) {
		workers  = new AliRdrWorker[in_num_threads];
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
		progress_clear [66] = '\r';
		progress_clear [67] = 0;

                load_references(in_p_refs);
	}
	
	~AliPool() {
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
		w_cmd.send_command(AliCmd::ALI_3D);
		
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

	void progress_start() {
		timer.tic();
		sprintf(progress_buffer,"        Aligning particles: Buffering...");
		int n = strlen(progress_buffer);
		progress_buffer[n]  = ' ';
		progress_buffer[65] = 0;
		printf(progress_buffer);
		fflush(stdout);
	}

	void show_progress(const int ptcls_in_tomo) {
		int cur_progress=0;
		while( (cur_progress=count_progress()) < ptcls_in_tomo ) {
			memset(progress_buffer,' ',66);
			if( cur_progress > 0 ) {
				int progress = count_accumul();
				float progress_percent = 100*(float)progress/float(n_ptcls);
				sprintf(progress_buffer,"        Aligning particles: %6.2f%%%%",progress_percent);
				int n = strlen(progress_buffer);
				add_etc(progress_buffer+n,progress,n_ptcls);
			}
			else {
				sprintf(progress_buffer,"        Aligning particles: Buffering...");
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
	
	void show_done() {
		memset(progress_buffer,' ',66);
		sprintf(progress_buffer,"        Aligning particles: 100.00%%%%");
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

#endif /// ALIGNER_H


