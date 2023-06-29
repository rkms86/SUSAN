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
#include "substack_crop.h"
#include "angles_provider.h"
#include "ref_ali.h"
#include "aligner_args.h"
#include "progress.h"

typedef enum {
	ALI_3D=1,
	ALI_2D
} AliCmd;

typedef enum {
	TM_NONE=0,
	TM_PYTHON,
	TM_MATLAB,
	TM_CSV
} TEMPLATE_MATCHING_OUTPUT;

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
    int K;
    int r_ix;
    int class_ix;
    int tomo_pos_x;
	int tomo_pos_y;
	int tomo_pos_z;

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

class TemplateMatchingReporter {
	
	int  tm_type;
	int  tm_dim;
	FILE *fp;
	
	float *c_cc;
	int   num_points;
	int   max_K;
	int   n_cc;
	float sigma;
	
public:
	TemplateMatchingReporter(int n_pts, int K, int dim) {
		tm_type    = TM_NONE;
		tm_dim     = dim;
		num_points = n_pts;
		max_K      = K;
		if(dim==2){       //2D alignment
			n_cc = n_pts * K;
		}
		else if (dim==3){ //3D alignment
			n_cc = n_pts;
		}
		c_cc       = new float[n_cc];
		sigma      = 0;
	}
	
	~TemplateMatchingReporter() {
		delete [] c_cc;
	}
	
	void start(int id,const char*type,const char*prefix,const float in_sigma) {
		sigma = in_sigma;
		
		if( strcmp(type,"none") == 0 ) {
			tm_type = TM_NONE;
		}
		else if( strcmp(type,"python") == 0 ) {
			tm_type = TM_PYTHON;
		}
		else if( strcmp(type,"matlab") == 0 ) {
			tm_type = TM_MATLAB;
		}
		else if( strcmp(type,"csv") == 0 ) {
			tm_type = TM_CSV;
		}
		
		if( tm_type == TM_PYTHON || tm_type == TM_MATLAB || tm_type == TM_CSV ) {
			char tm_file[SUSAN_FILENAME_LENGTH];
			sprintf(tm_file,"%s_worker%02d.txt",prefix,id);
			fp = fopen(tm_file,"w");
			if( tm_type == TM_CSV ) {
				if (tm_dim == 2){
					fprintf(fp,"TID,PartID,RID,ProjID,X,Y,CC\n");
				}
				else if (tm_dim == 3){
					fprintf(fp,"TID,PartID,RID,X,Y,Z,CC\n");
				}
			}
		}
	}
	
	void finish() {
		if( tm_type == TM_PYTHON || tm_type == TM_MATLAB || tm_type == TM_CSV ) {
			fclose(fp);
		}
	}

	void clear_cc() {
		memset(c_cc,0,n_cc*sizeof(float));
	}

	void push_cc(const float*p_cc) {
		memcpy(c_cc,p_cc,n_cc*sizeof(float));
	}
	
	void save_cc(int tid,int rid,int pid,int tx,int ty,int tz,const Vec3*c_pts) {
		if( tm_type != TM_NONE ) {
			int x,y,z;
			
			int proj_id, point_id;
			
			float avg,std;
			Math::get_avg_std(avg,std,c_cc,n_cc);

			for(int cc_index=0;cc_index<n_cc;cc_index++){
				if (tm_dim == 2){
					proj_id  = cc_index / num_points;
					point_id = cc_index % num_points;
				}
				else if (tm_dim == 3){
					proj_id  = 0;
					point_id = cc_index;
				}

				x = (int)roundf(c_pts[point_id].x);
                y = (int)roundf(c_pts[point_id].y);
                z = (int)roundf(c_pts[point_id].z);
				
				if ( ((sigma > 0) && (c_cc[cc_index] > avg+sigma*std)) || (sigma <= 0) ){
					if (tm_dim == 2){
						if( tm_type == TM_PYTHON )
                                fprintf(fp,"cc_tomo%03d_ptcl%d_ref%02d_proj%02d[%d,%d] = %f\n", tid, pid, rid, proj_id, x, y, c_cc[cc_index]);
                        else if( tm_type == TM_MATLAB )
                                fprintf(fp,"cc_tomo%03d_ptcl%d_ref%02d_proj%02d(%d,%d) = %f;\n",tid, pid, rid, proj_id, (x+1), (y+1), c_cc[cc_index]);
                        else if( tm_type == TM_CSV )
                                fprintf(fp,"%d,%d,%d,%d,%d,%d,%f\n", tid, pid, rid, proj_id, x, y, c_cc[cc_index]);
					}
					else if (tm_dim == 3){
						if( tm_type == TM_PYTHON )
                                fprintf(fp,"cc_tomo%03d_ptcl%d_ref%02d[%d,%d,%d] = %f\n", tid, pid, rid, (z+tz),  (y+ty),  (x+tx),  c_cc[cc_index]);
                        else if( tm_type == TM_MATLAB )
                                fprintf(fp,"cc_tomo%03d_ptcl%d_ref%02d(%4d,%4d,%4d) = %f;\n",tid, pid, rid, (x+tx+1), (y+ty+1), (z+tz+1), c_cc[cc_index]);
                        else if( tm_type == TM_CSV )
                                fprintf(fp,"%d,%d,%d,%d,%d,%d,%f\n",tid, pid, rid, (x+tx), (y+ty), (z+tz), c_cc[cc_index]);
					}
				}
				
			}
		}
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

    uint32 cc_stats;

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
    
    const char *tm_type;
    const char *tm_prefix;
    int         tm_dim;
    float       tm_sigma;

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

        AliData ali_data(MP,NP,max_K,off_par,off_type,stream);
        
        TemplateMatchingReporter tm_rep(ali_data.n_pts,max_K,tm_dim);
        tm_rep.start(worker_id,tm_type,tm_prefix,tm_sigma);

        RadialAverager rad_avgr(M,N,max_K);

        GPU::GArrSingle ctf_wgt;
        ctf_wgt.alloc(MP*NP*max_K);

        int num_vols = R;
        if( ali_halves ) num_vols = 2*R;
        AliRef*vols = new AliRef[num_vols];
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
                    align3D(vols,ctf_wgt,ss_data,ali_data,rad_avgr,tm_rep,stream);
                    break;
                case ALI_2D:
                    align2D(vols,ctf_wgt,ss_data,ali_data,rad_avgr,tm_rep,stream);
                    break;
                default:
                    break;
            }
        }

        GPU::sync();
        tm_rep.finish();
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

    void align3D(AliRef*vols,GPU::GArrSingle&ctf_wgt,AliSubstack&ss_data,AliData&ali_data,RadialAverager&rad_avgr,TemplateMatchingReporter&tm_rep,GPU::Stream&stream) {
        p_buffer->RO_sync();
        while( p_buffer->RO_get_status() > DONE ) {
            if( p_buffer->RO_get_status() == READY ) {
                AliBuffer*ptr = (AliBuffer*)p_buffer->RO_get_buffer();
                create_ctf(ctf_wgt,ptr,stream);
                add_data(ss_data,ctf_wgt,ptr,rad_avgr,stream);
                add_rec_weight(ss_data,ptr,stream);
                angular_search_3D(vols[ptr->r_ix],ss_data,ctf_wgt,ptr,ali_data,rad_avgr,tm_rep,stream);
                stream.sync();
                set_classification(ptr);
            }
            p_buffer->RO_sync();
        }
    }

    void align2D(AliRef*vols,GPU::GArrSingle&ctf_wgt,AliSubstack&ss_data,AliData&ali_data,RadialAverager&rad_avgr,TemplateMatchingReporter&tm_rep,GPU::Stream&stream) {
        p_buffer->RO_sync();
        while( p_buffer->RO_get_status() > DONE ) {
            if( p_buffer->RO_get_status() == READY ) {
                AliBuffer*ptr = (AliBuffer*)p_buffer->RO_get_buffer();
                create_ctf(ctf_wgt,ptr,stream);
                add_data(ss_data,ctf_wgt,ptr,rad_avgr,stream);
                angular_search_2D(vols[ptr->r_ix],ss_data,ctf_wgt,ptr,ali_data,rad_avgr,tm_rep,stream);
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

    void add_data(AliSubstack&ss_data,GPU::GArrSingle&ctf_wgt,AliBuffer*ptr,RadialAverager&rad_avgr,GPU::Stream&stream) {

        switch( pad_type ) {
            case PAD_ZERO:
                ss_data.pad_zero(stream);
                break;
            case PAD_GAUSSIAN:
                ss_data.pad_normal(ptr->g_pad,ptr->K,stream);
                break;
            default:
                break;
        }

        ss_data.add_data(ptr->g_stk,ptr->g_ali,ptr->K,stream);

        switch( ctf_type ) {
            case ALI_ON_SUBSTACK:
                ss_data.correct_wiener(ptr->ctf_vals,ctf_wgt,ptr->g_def,bandpass,ptr->K,stream);
                break;
            case ALI_ON_SUBSTACK_SSNR:
                ss_data.correct_wiener_ssnr(ptr->ctf_vals,ctf_wgt,ptr->g_def,bandpass,ssnr,ptr->K,stream);
                break;
            case ALI_CUMULATIVE_FSC:
                rad_avgr.preset_FRC(ss_data.ss_fourier,ptr->K,stream);
                break;
            default:
                break;
        }
    }

    void add_rec_weight(AliSubstack&ss_data,AliBuffer*ptr,GPU::Stream&stream) {
        float w_total=0;
        for(int k=0;k<ptr->K;k++) {
            w_total += ptr->c_ali.ptr[k].w;
        }
        ss_data.apply_radial_wgt(w_total,ptr->K,stream);
    }

    void angular_search_3D(AliRef&vol,AliSubstack&ss_data,GPU::GArrSingle&ctf_wgt,AliBuffer*ptr,AliData&ali_data,RadialAverager&rad_avgr,TemplateMatchingReporter&tm_rep,GPU::Stream&stream) {

        Rot33 Rot;
        single max_cc=0,cc;
        int max_idx=0,idx;
        M33f max_R,R_ite,R_tmp;
        M33f R_lvl = Eigen::MatrixXf::Identity(3,3);

        single running_avg = 0;
        single running_std = 0;
        single running_cnt = 0;

        for( ang_prov.levels_init(); ang_prov.levels_available(); ang_prov.levels_next() ) {
            for( ang_prov.sym_init(); ang_prov.sym_available(); ang_prov.sym_next() ) {
                for( ang_prov.cone_init(); ang_prov.cone_available(); ang_prov.cone_next() ) {
                    for( ang_prov.inplane_init(); ang_prov.inplane_available(); ang_prov.inplane_next() ) {

                        ang_prov.get_current_R(R_ite);
                        R_tmp = R_ite*R_lvl;
                        Math::set(Rot,R_tmp);
                        ali_data.rotate_post(Rot,ptr->g_ali,ptr->K,stream);
                        ali_data.project(vol.ref,bandpass,ptr->K,stream);

                        if( ctf_type == ALI_ON_REFERENCE )
                            ali_data.multiply(ctf_wgt,ptr->K,stream);

                        if( ctf_type == ALI_CUMULATIVE_FSC ) {
                            ali_data.multiply(ctf_wgt,ptr->K,stream);
                            rad_avgr.preset_FRC(ali_data.prj_c,ptr->K,stream);
                        }

                        ali_data.multiply(ss_data.ss_fourier,ptr->K,stream);

                        ali_data.apply_bandpass(bandpass,ptr->K,stream);
                        ali_data.invert_fourier(ptr->K,stream);

                        if( ctf_type == ALI_ON_REFERENCE ||
                            ctf_type == ALI_ON_SUBSTACK )
                        {
                            ali_data.scale(NP*NP,ptr->K,stream);
                            ali_data.scale(NP*NP,ptr->K,stream);
                        }
                        else {
                            ali_data.scale(4*float(NP*NP)/M_PI,ptr->K,stream);
                        }
                        
                        ali_data.sparse_reconstruct(ptr->g_ali,ptr->K,stream);
                        stream.sync();
                        ali_data.get_max_cc(cc,idx,ali_data.c_cc);
                        
                        if( cc > max_cc ) {
                            max_idx = idx;
                            max_cc  = cc;
                            max_R   = R_tmp;
                            tm_rep.push_cc(ali_data.c_cc);
                        }
                        if( cc_stats == CC_SIGMA ) {
                            running_avg += cc;
                            running_std += (cc*cc);
                            running_cnt += 1;
                        }
                        if( cc_stats == CC_PROB ) {
                            running_avg += ali_data.get_sum_cc(ali_data.c_cc);
                        }
                        
                    } // INPLANE
                } // CONE
            } // SYMMETRY
            R_lvl = max_R;
        } // REFINE

        if( cc_stats == CC_SIGMA ) {
            if( (running_cnt > 0) || (running_std<SUSAN_FLOAT_TOL) ) {
                running_avg = running_avg/running_cnt;
                running_std = running_std/running_cnt;
                running_std = running_std - (running_avg*running_avg);
                running_std = sqrtf(running_std);
                max_cc = (max_cc - running_avg) / running_std;
                max_cc = fmax(max_cc,0.0);
            }
            else {
                max_cc  = 0;
                max_idx = (ali_data.n_pts/2);
            }
        }
        if( cc_stats == CC_PROB ) {
            max_cc = max_cc/running_avg;
        }
        
        update_particle_3D(ptr->ptcl,max_R,ali_data.c_pts[max_idx],max_cc,ptr->class_ix,ptr->ctf_vals.apix);
        tm_rep.save_cc(ptr->ptcl.tomo_id(),ptr->ptcl.ref_cix()+1,ptr->ptcl.ptcl_id(),ptr->tomo_pos_x,ptr->tomo_pos_y,ptr->tomo_pos_z,ali_data.c_pts);
    }

    void angular_search_2D(AliRef&vol,AliSubstack&ss_data,GPU::GArrSingle&ctf_wgt,AliBuffer*ptr,AliData&ali_data,RadialAverager&rad_avgr,TemplateMatchingReporter&tm_rep,GPU::Stream&stream) {
        
	ang_prov.refine_level = 0;
        Rot33 Rot;
        M33f  R_ite;

        single max_cc[ptr->K];
        single ite_cc[ptr->K];
        int    max_idx[ptr->K];
        int    ite_idx[ptr->K];
        M33f   max_R[ptr->K];
        // single := float //
        single cc_placeholder[(ptr->K)*(ali_data.n_pts)];

        memset(max_cc,         0, sizeof(single)*ptr->K);
        memset(max_idx,        0, sizeof(single)*ptr->K);
        memset(cc_placeholder, 0, sizeof(single)*(ptr->K)*(ali_data.n_pts));

        for( ang_prov.levels_init(); ang_prov.levels_available(); ang_prov.levels_next() ) {
            for( ang_prov.sym_init(); ang_prov.sym_available(); ang_prov.sym_next() ) {
                for( ang_prov.cone_init(); ang_prov.cone_available(); ang_prov.cone_next() ) {
                    for( ang_prov.inplane_init(); ang_prov.inplane_available(); ang_prov.inplane_next() ) {

                        ang_prov.get_current_R(R_ite);
                        Math::set(Rot,R_ite);
                        ali_data.rotate_pre(Rot,ptr->g_ali,ptr->K,stream);
                        ali_data.project(vol.ref,bandpass,ptr->K,stream);

                        if( ctf_type == ALI_ON_REFERENCE )
                            ali_data.multiply(ctf_wgt,ptr->K,stream);

                        if( ctf_type == ALI_CUMULATIVE_FSC ) {
                            ali_data.multiply(ctf_wgt,ptr->K,stream);
                            rad_avgr.preset_FRC(ali_data.prj_c,ptr->K,stream);
                        }

                        ali_data.multiply(ss_data.ss_fourier,ptr->K,stream);

                        ali_data.apply_bandpass(bandpass,ptr->K,stream);
                        ali_data.invert_fourier(ptr->K,stream);
                        stream.sync();

                        if( ctf_type == ALI_ON_REFERENCE ||
                            ctf_type == ALI_ON_SUBSTACK )
                        {
                            ali_data.scale(N*N,ptr->K,stream);
                            ali_data.scale(N*N,ptr->K,stream);
                        }
                        else {
                            ali_data.scale(4*float(NP*NP)/M_PI,ptr->K,stream);
                        }

                        ali_data.extract_cc(ite_cc,ite_idx,ptr->g_ali,ptr->K,stream);

                        for(int i=0;i<ptr->K;i++) {
                            if( ite_cc[i] > max_cc[i] ) {
                                max_idx[i] = ite_idx[i];
                                max_cc[i]  = ite_cc[i];
                                max_R[i]   = R_ite;
				// cc_placeholder stores cc-map (offset->cc) only for the best orientation 
				for(int j=0;j<ali_data.n_pts;j++){
					cc_placeholder[i*ali_data.n_pts + j] = ali_data.c_cc[i*ali_data.n_pts + j];
                }
                            }
                        }
                    } // INPLANE
                } // CONE
            } // SYMMETRY
        } // REFINE

	tm_rep.push_cc(cc_placeholder);

        single cc_acc=0,wgt_acc=0;
        for(int i=0;i<ptr->K;i++) {
            update_particle_2D(ptr->ptcl,max_R[i],ali_data.c_pts[max_idx[i]],max_cc[i],i,ptr->ctf_vals.apix);
            if( max_cc[i] > 0 && ptr->ptcl.prj_w[i] > 0 ) {
                cc_acc += max_cc[i];
                wgt_acc += ptr->ptcl.prj_w[i];
            }
        }
        ptr->ptcl.ali_cc[ptr->class_ix] = cc_acc/max(wgt_acc,1.0);
	tm_rep.save_cc(ptr->ptcl.tomo_id(),ptr->ptcl.ref_cix()+1,ptr->ptcl.ptcl_id(),ptr->tomo_pos_x,ptr->tomo_pos_y,ptr->tomo_pos_z,ali_data.c_pts);
    }

    void update_particle_3D(Particle&ptcl,const M33f&Rot,const Vec3&t,const single cc, const int ref_ix,const float apix) {

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

            if( drift3D ) {
                ptcl.ali_t[ref_ix].x += tt(0)*apix;
                ptcl.ali_t[ref_ix].y += tt(1)*apix;
                ptcl.ali_t[ref_ix].z += tt(2)*apix;
            }
            else {
                ptcl.ali_t[ref_ix].x = tt(0)*apix;
                ptcl.ali_t[ref_ix].y = tt(1)*apix;
                ptcl.ali_t[ref_ix].z = tt(2)*apix;
            }
        }
    }

    void update_particle_2D(Particle&ptcl,const M33f&Rot,const Vec3&t,const single cc, const int prj_ix,const float apix) {

        if( ptcl.prj_w[prj_ix] > 0 && cc > 0 ) {
            ptcl.prj_cc[prj_ix] = cc;

            M33f Rprv;
            V3f eu_prv;
            eu_prv(0) = ptcl.prj_eu[prj_ix].x;
            eu_prv(1) = ptcl.prj_eu[prj_ix].y;
            eu_prv(2) = ptcl.prj_eu[prj_ix].z;
            Math::eZYZ_Rmat(Rprv,eu_prv);
            M33f Rnew = Rot*Rprv;
            Math::Rmat_eZYZ(eu_prv,Rnew);
            ptcl.prj_eu[prj_ix].x = eu_prv(0);
            ptcl.prj_eu[prj_ix].y = eu_prv(1);
            ptcl.prj_eu[prj_ix].z = eu_prv(2);

            V3f tp,tt;
            tp(0) = t.x;
            tp(1) = t.y;
            tp(2) = t.z;
            tt = Rot.transpose()*tp;

            if( drift2D ) {
                ptcl.prj_t[prj_ix].x += tt(0)*apix;
                ptcl.prj_t[prj_ix].y += tt(1)*apix;
            }
            else {
                ptcl.prj_t[prj_ix].x = tt(0)*apix;
                ptcl.prj_t[prj_ix].y = tt(1)*apix;
            }
        }
    }

    void set_classification(AliBuffer*ptr) {
        if( ptr->class_ix+1 == ptr->ptcl.n_refs ) {
            float max_cc = ptr->ptcl.ali_cc[0];
            ptr->ptcl.ref_cix() = 0;
            for(int i=0;i<ptr->ptcl.n_refs;i++) {
                if( max_cc < ptr->ptcl.ali_cc[i] ) {
                    max_cc = ptr->ptcl.ali_cc[i];
                    ptr->ptcl.ref_cix() = i;
                }
            }
        }
    }

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
        // info->n_threads = number_of_gpus * threads_per_gpu, in order to get a gpu index from a worker index we have to make an integer division by threads_per_gpu instead of remainder of the devision by info->n_threads, which will not change id, since id < info->n_threads all the time.
        int threads_per_gpu = (info->n_threads) / (info->n_gpu);
        gpu_ix   = info->p_gpu[ id / threads_per_gpu ];
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
        gpu_worker.cc_stats   = p_info->cc_stats;
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
        gpu_worker.tm_type    = p_info->tm_type;
        gpu_worker.tm_prefix  = p_info->tm_pfx;
        gpu_worker.tm_dim     = p_info->type;
        gpu_worker.tm_sigma   = p_info->tm_sigma;
        gpu_worker.start();
    }

    void crop_loop(DoubleBufferHandler&stack_buffer,GPU::Stream&stream) {
        stack_buffer.WO_sync(EMPTY);
        for(int i=worker_id;i<p_ptcls->n_ptcl;i+=p_info->n_threads) {
            work_progress++;
            work_accumul++;
            for(int r=0;r<R;r++) {
                AliBuffer*ptr = (AliBuffer*)stack_buffer.WO_get_buffer();
                p_ptcls->get(ptr->ptcl,i);
                read_defocus(ptr);
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

        memcpy( (void**)(ptr->c_def.ptr), (const void**)(ptr->ptcl.def), sizeof(Defocus)*ptr->K  );

        for(int k=0;k<ptr->K;k++) {
            if( ptr->c_def.ptr[k].max_res > 0 ) {
                ptr->c_def.ptr[k].max_res = ((float)NP)*p_tomo->pix_size/ptr->c_def.ptr[k].max_res;
                ptr->c_def.ptr[k].max_res = min(ptr->c_def.ptr[k].max_res+bp_pad,(float)NP/2);
            }
        }
    }

    void crop_substack(AliBuffer*ptr,const int r) {
        V3f pt_tomo,pt_stack,pt_crop,pt_subpix,eu_ZYZ;
        M33f R_tmp,R_ali,R_stack,R_gpu;

        ptr->class_ix = r;

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
        
        ptr->tomo_pos_x = (int)roundf( (pt_tomo(0)/p_tomo->pix_size) + p_tomo->tomo_center(0) );
        ptr->tomo_pos_y = (int)roundf( (pt_tomo(1)/p_tomo->pix_size) + p_tomo->tomo_center(1) );
        ptr->tomo_pos_z = (int)roundf( (pt_tomo(2)/p_tomo->pix_size) + p_tomo->tomo_center(2) );

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
                if( drift2D ) {
                    pt_crop(0) += ptr->ptcl.prj_t[k].x;
                    pt_crop(1) += ptr->ptcl.prj_t[k].y;
                }

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
                    float avg,std;
                    float *ss_ptr = ptr->c_stk.ptr+(k*N*N);
                    Math::get_avg_std(avg,std,ss_ptr,N*N);

                    if( std < SUSAN_FLOAT_TOL || isnan(std) || isinf(std) ) {
                        ptr->c_pad.ptr[k].x = 0;
                        ptr->c_pad.ptr[k].y = 1;
                        ptr->c_ali.ptr[k].w = 0;
                    }
                    else {
                        if( p_info->norm_type == NO_NORM ) {
                            ptr->c_pad.ptr[k].x = avg;
                            ptr->c_pad.ptr[k].y = std;
                        }
                        else {
                            Math::normalize(ss_ptr,N*N,avg,std);
                            ptr->c_pad.ptr[k].x = 0;
                            if( p_info->norm_type == ::ZERO_MEAN ) {
                                ptr->c_pad.ptr[k].y = std;
                            }
                            if( p_info->norm_type == ZERO_MEAN_1_STD ) {
                                ptr->c_pad.ptr[k].y = 1;
                            }
                            if( p_info->norm_type == ZERO_MEAN_W_STD ) {
                                ptr->c_pad.ptr[k].y = ptr->ptcl.prj_w[k];
                            }
                        }
                    }
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

    ProgressReporter progress;

    AliPool(ArgsAli::Info*info,References*in_p_refs,int in_max_K,int num_ptcls,StackReader&stkrdr,int in_num_threads)
     : PoolCoordinator(stkrdr,in_num_threads),
       w_cmd(2*in_num_threads+1),
       progress("    Aligning particles",num_ptcls)
    {
        workers  = new AliRdrWorker[in_num_threads];
        p_info   = info;
        max_K    = in_max_K;
        n_ptcls  = num_ptcls;
        N = info->box_size;
        M = (N/2)+1;
        P = info->pad_size;
        NP = N+P;
        MP = (NP/2)+1;
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
        if( p_info->type == 3 )
            w_cmd.send_command(AliCmd::ALI_3D);
        if( p_info->type == 2 )
            w_cmd.send_command(AliCmd::ALI_2D);

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
        progress.start();
    }

    virtual void show_progress(const int ptcls_in_tomo) {
        int cur_progress=0;
        while( (cur_progress=count_progress()) < ptcls_in_tomo ) {
            int total_progress = count_accumul();
            progress.update(total_progress,cur_progress==0);
            sleep(2);
        }
    }
	
    virtual void show_done() {
        progress.finish();
    }

};

#endif /// ALIGNER_H


