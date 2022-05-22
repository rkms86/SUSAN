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

#ifndef RECONSTRUCT_ARGS_H
#define RECONSTRUCT_ARGS_H

#include <pthread.h>
#include <getopt.h>
#include "datatypes.h"
#include "gpu.h"

#include "angles_symmetry.h"

namespace ArgsRec {

typedef enum {
	NO_NORM=0,
	ZERO_MEAN,
	ZERO_MEAN_W_STD,
	ZERO_MEAN_1_STD
} NormalizationType_t;

typedef enum {
	PAD_ZERO=0,
	PAD_GAUSSIAN
} PaddingType_t;

typedef enum {
	NO_INV=0,
	PHASE_FLIP,
	WIENER_INV,
	WIENER_INV_SSNR
} InversionType_t;

typedef struct {
	int    n_gpu;
    uint32 p_gpu[SUSAN_MAX_N_GPU];
    uint32 n_threads;
	uint32 box_size;
    single fpix_min;
    single fpix_max;
    single fpix_roll;
    uint32 pad_size;
    uint32 pad_type;
    uint32 ctf_type;
    uint32 norm_type;
    int    w_inv_ite;
    float  w_inv_std;
    float  ssnr_F;
    float  ssnr_S;
    bool   rec_halves;
    
    char   sym[64];
    
    char   out_pfx[SUSAN_FILENAME_LENGTH];
    char   ptcls_in[SUSAN_FILENAME_LENGTH];
    char   tomos_in[SUSAN_FILENAME_LENGTH];
    
    int verbosity;
} Info;

uint32 get_pad_type(const char*arg) {
	uint32 rslt = PAD_ZERO;
	bool all_ok = false;
	
	if( strcmp(arg,"zero") == 0 ) {
		rslt = PAD_ZERO;
		all_ok = true;
	}
	
	if( strcmp(arg,"noise") == 0 ) {
		rslt = PAD_GAUSSIAN;
		all_ok = true;
	}
	
	if( !all_ok ) {
		fprintf(stderr,"Invalid padding type %s. Options are: zero or noise. Defaulting to zero.\n",arg);
	}
	
	return rslt;
}

uint32 get_norm_type(const char*arg) {
	uint32 rslt = NO_NORM;
	bool all_ok = false;
	
	if( strcmp(arg,"none") == 0 ) {
		rslt = PAD_ZERO;
		all_ok = true;
	}
	
	if( strcmp(arg,"zero_mean") == 0 ) {
		rslt = ZERO_MEAN;
		all_ok = true;
	}
	
	if( strcmp(arg,"zero_mean_proj_weight") == 0 ) {
		rslt = ZERO_MEAN_W_STD;
		all_ok = true;
	}
	
	if( strcmp(arg,"zero_mean_one_std") == 0 ) {
		rslt = ZERO_MEAN_1_STD;
		all_ok = true;
	}
	
	if( !all_ok ) {
		fprintf(stderr,"Invalid normalization type %s. Options are: none, zero_mean, zero_mean_proj_weight and zero_mean_one_std. Defaulting to none.\n",arg);
	}
	
	return rslt;
}

uint32 get_ctf_type(const char*arg) {
	uint32 rslt = WIENER_INV;
	bool all_ok = false;
	
	if( strcmp(arg,"none") == 0 ) {
		rslt = NO_INV;
		all_ok = true;
	}
	
	if( strcmp(arg,"phase_flip") == 0 ) {
		rslt = PHASE_FLIP;
		all_ok = true;
	}
	
	if( strcmp(arg,"wiener") == 0 ) {
		rslt = WIENER_INV;
		all_ok = true;
	}
	
	if( strcmp(arg,"wiener_ssnr") == 0 ) {
		rslt = WIENER_INV_SSNR;
		all_ok = true;
	}
	
	if( !all_ok ) {
		fprintf(stderr,"Invalid ctf correction type %s. Options are: none, phase_flip, wiener and wiener_ssnr. Defaulting to wiener.\n",arg);
	}
	
	return rslt;
}

void set_symmetry(Info&info,const char*arg) {
	
	uint32 num_angs;
	M33f*p_angs;
	
	p_angs = AnglesSymmetry::get_rotation_list(num_angs,arg);

	if( num_angs == 0  ) {
		strcpy(info.sym,"c1");
		fprintf(stderr,"Invalid symmetry option %s. Supported: c1, c2, cXXX... Defaulting to c1.\n",arg);
	}
	else {
		strcpy(info.sym,arg);
		delete [] p_angs;
	}
}

bool validate(const Info&info) {
	bool rslt = true;
	if( info.fpix_min >= info.fpix_max ) {
		fprintf(stderr,"Invalid bandpass range: %f - %f.\n",info.fpix_min,info.fpix_max);
		rslt = false;
	}
	if( !IO::exists(info.ptcls_in) ) {
		fprintf(stderr,"Particles file %s does not exist.\n",info.ptcls_in);
		rslt = false;
	}
	if( !IO::exists(info.tomos_in) ) {
		fprintf(stderr,"Tomos file %s does not exist.\n",info.tomos_in);
		rslt = false;
	}
	if( strlen(info.out_pfx) == 0 ) {
		fprintf(stderr,"Output pfx missing.\n");
		rslt = false;
	}
	if( info.n_gpu < 1 ) {
		fprintf(stderr,"At least 1 GPU must be requested.\n");
		rslt = false;
	}
	else {
		int available_gpus = GPU::count_devices();
		if(available_gpus==0) {
			fprintf(stderr,"Not available GPUs on the system.\n");
			rslt = false;
		}
		else {
			for(int i=0;i<info.n_gpu;i++) {
				if( info.p_gpu[i] >= available_gpus ) {
					fprintf(stderr,"Requesting unavalable GPU with ID %d.\n",info.p_gpu[i]);
					rslt = false;
				}
			}
		}
	}
	return rslt;
};

bool parse_args(Info&info,int ac,char** av) {
	/// Default values:
	info.n_gpu      = 0;
    info.n_threads  = 1;
	info.box_size   = 200;
    info.fpix_min   = 0;
    info.fpix_max   = 30;
    info.fpix_roll  = 4;
    info.pad_size   = 0;
    info.pad_type   = PAD_ZERO;
    info.ctf_type   = WIENER_INV_SSNR;
    info.norm_type  = NO_NORM;
    info.w_inv_ite  = 10;
    info.w_inv_std  = 0.75;
    info.ssnr_F     = 0;
    info.ssnr_S     = 1;
    info.rec_halves = false;
    info.verbosity  = 0;
    memset(info.p_gpu   ,0,SUSAN_MAX_N_GPU*sizeof(uint32));
	memset(info.out_pfx ,0,SUSAN_FILENAME_LENGTH*sizeof(char));
	memset(info.ptcls_in,0,SUSAN_FILENAME_LENGTH*sizeof(char));
	memset(info.tomos_in,0,SUSAN_FILENAME_LENGTH*sizeof(char));
	strcpy(info.sym,"c1");
	
	/// Parse inputs:
	enum {
		TOMOS_FILE,
        OUT_PREFIX,
        PTCLS_FILE,
        N_THREADS,
        GPU_LIST,
        BOX_SIZE,
        PAD_SIZE,
        PAD_TYPE,
        NORM_TYPE,
        CTF_TYPE,
        SSNR,
        W_INV_ITE,
        W_INV_STD,
        BANDPASS,
        ROLLOFF_F,
        SYMMETRY,
        VERBOSITY,
        REC_HALVES
    };

    int c;
    static struct option long_options[] = {
        {"tomos_file",  1, 0, TOMOS_FILE},
        {"out_prefix",  1, 0, OUT_PREFIX},
        {"ptcls_file",  1, 0, PTCLS_FILE},
        {"n_threads",   1, 0, N_THREADS },
        {"gpu_list",    1, 0, GPU_LIST  },
        {"box_size",    1, 0, BOX_SIZE  },
        {"pad_size",    1, 0, PAD_SIZE  },
        {"pad_type",    1, 0, PAD_TYPE  },
        {"norm_type",   1, 0, NORM_TYPE },
        {"ctf_type",    1, 0, CTF_TYPE  },
        {"ssnr_param",  1, 0, SSNR      },
        {"w_inv_iter",  1, 0, W_INV_ITE },
        {"w_inv_gstd",  1, 0, W_INV_STD },
        {"bandpass",    1, 0, BANDPASS  },
        {"rolloff_f",   1, 0, ROLLOFF_F },
        {"symmetry",    1, 0, SYMMETRY  },
        {"rec_halves",  1, 0, REC_HALVES},
        {"verbosity",   1, 0, VERBOSITY },
        {0, 0, 0, 0}
    };
    
    single *tmp_single;
    uint32 *tmp_uint32;
    float  tmp;
    while( (c=getopt_long_only(ac, av, "", long_options, 0)) >= 0 ) {
        switch(c) {
			case TOMOS_FILE:
				strcpy(info.tomos_in,optarg);
				break;
			case OUT_PREFIX:
				strcpy(info.out_pfx,optarg);
				break;
			case PTCLS_FILE:
				strcpy(info.ptcls_in,optarg);
				break;
			case BOX_SIZE:
				info.box_size = atoi(optarg);
				tmp = (float)(info.box_size);
				info.box_size = (int)(2.0*roundf(tmp/2)); // Force box to be multiple of 2.
				break;
			case N_THREADS:
				info.n_threads = atoi(optarg);
				break;
			case GPU_LIST:
				info.n_gpu = IO::parse_uint32_strlist(tmp_uint32, optarg);
				if( info.n_gpu > SUSAN_MAX_N_GPU ) {
					fprintf(stderr,"Requesting %d GPUs. Maximum is %d\n",info.n_gpu,SUSAN_MAX_N_GPU);
					exit(1);
				}
				memcpy(info.p_gpu,tmp_uint32,info.n_gpu*sizeof(uint32));
				delete [] tmp_uint32;
				break;
			case PAD_SIZE:
				info.pad_size = atoi(optarg);
				tmp = (float)(info.pad_size);
				info.pad_size = (int)(2.0*roundf(tmp/2)); // Force pad to be multiple of 2.
				break;
			case PAD_TYPE:
				info.pad_type = get_pad_type(optarg);
				break;
			case NORM_TYPE:
				info.norm_type = get_norm_type(optarg);
				break;
			case CTF_TYPE:
				info.ctf_type = get_ctf_type(optarg);
				break;
			case BANDPASS:
				IO::parse_single_strlist(tmp_single, optarg);
				info.fpix_min = tmp_single[0];
				info.fpix_max = tmp_single[1];
				delete [] tmp_single;
				break;
			case ROLLOFF_F:
				info.fpix_roll = atof(optarg);
				break;
			case SSNR:
				IO::parse_single_strlist(tmp_single, optarg);
				info.ssnr_F = tmp_single[0];
				info.ssnr_S = tmp_single[1];
				delete [] tmp_single;
				break;
			case W_INV_ITE:
				info.w_inv_ite = atoi(optarg);
				break;
			case W_INV_STD:
				info.w_inv_std = atof(optarg);
				break;
			case SYMMETRY:
				set_symmetry(info,optarg);
				break;
			case REC_HALVES:
				info.rec_halves = (atoi(optarg)>0);
				break;
            case VERBOSITY:
                info.verbosity = atoi(optarg);
                break;
			default:
				printf("Unknown parameter %d\n",c);
				exit(1);
				break;
        } /// switch
    } /// while(c)
    
    return validate(info);
}

void print_full(const Info&info,FILE*fp) {
	fprintf(fp,"\tVolume reconstruction");
	if( info.rec_halves )
		fprintf(fp," (including half-sets)");
	fprintf(fp,":\n");

	fprintf(fp,"\t\tParticles file: %s.\n",info.ptcls_in);
	fprintf(fp,"\t\tTomograms file: %s.\n",info.tomos_in);
	fprintf(fp,"\t\tOutput prefix: %s.\n",info.out_pfx);

    fprintf(fp,"\t\tVolume size: %dx%dx%d",info.box_size,info.box_size,info.box_size);
    if( info.pad_size > 0 ) {
        fprintf(fp,", with padding of %d voxels",info.pad_size);
    }
    fprintf(fp,".\n");
    
    if( info.n_gpu > 1 ) {
        fprintf(fp,"\t\tUsing %d GPUs (GPU ids: %d",info.n_gpu,info.p_gpu[0]);
        for(int i=1;i<info.n_gpu;i++)
            fprintf(fp,",%d",info.p_gpu[i]);
        fprintf(fp,"), ");
    }
    else {
        fprintf(fp,"\t\tUsing 1 GPU (GPU id: %d), ",info.p_gpu[0]);
    }
    
    if( info.n_threads > 1 ) {
		fprintf(fp,"and %d threads.\n",info.n_threads);
	}
	else{
		fprintf(fp,"and 1 thread.\n");
	}
	
	fprintf(fp,"\t\tBandpass: [%.1f - %.1f] fourier pixels",info.fpix_min,info.fpix_max);
	if( info.fpix_roll > 0 )
		fprintf(fp," with a roll off of %.2f.\n",info.fpix_roll);
	else
		fprintf(fp,".\n");

	if( info.pad_size > 0 ) {
		if( info.pad_type == PAD_ZERO )
			fprintf(fp,"\t\tPadding policy: Fill with zeros.\n");
		if( info.pad_type == PAD_GAUSSIAN )
			fprintf(fp,"\t\tPadding policy: Fill with gaussian noise.\n");
	}
	
    if( info.ctf_type == NO_INV )
		fprintf(fp,"\t\tCTF correction policy: Disabled.\n");
	if( info.ctf_type == PHASE_FLIP )
		fprintf(fp,"\t\tCTF correction policy: Phase-flip.\n");
	if( info.ctf_type == WIENER_INV )
		fprintf(fp,"\t\tCTF correction policy: Wiener inversion.\n");
	if( info.ctf_type == WIENER_INV_SSNR )
		fprintf(fp,"\t\tCTF correction policy: Wiener inversion with SSNR(f) = (100^(3*%.2f))*e^(-100*%.2f*f).\n",info.ssnr_S,info.ssnr_F);
	
	fprintf(fp,"\t\tInversion of the sampled fourier space using %d iterations and a gaussian filter with std of %f.\n",info.w_inv_ite,info.w_inv_std);
	
	if( info.norm_type == NO_NORM )
		fprintf(fp,"\t\tSubstack normalization policy: Disabled.\n");
	if( info.norm_type == ZERO_MEAN )
		fprintf(fp,"\t\tSubstack normalization policy: Mean=0.\n");
	if( info.norm_type == ZERO_MEAN_1_STD )
		fprintf(fp,"\t\tSubstack normalization policy: Mean=0, Std=1.\n");
	if( info.norm_type == ZERO_MEAN_W_STD )
		fprintf(fp,"\t\tSubstack normalization policy: Mean=0, Std according to projection weight.\n");
	
	fprintf(fp,"\t\tSymmetry type: %s.\n",info.sym);
}

void print_minimal(const Info&info,FILE*fp) {
	fprintf(fp,"    Volume reconstruction");
	if( info.rec_halves )
		fprintf(fp," (including half-sets)");
	
    fprintf(fp,". Box size: %d",info.box_size);
    if( info.pad_size > 0 ) {
        fprintf(fp," + %d (pad)",info.pad_size);
    }
    fprintf(fp,"\n");

    fprintf(fp,"    - Input files: %s | %s\n",info.ptcls_in,info.tomos_in);
    fprintf(fp,"    - Output prefix: %s\n",info.out_pfx);
    
    if( info.n_gpu > 1 ) {
        fprintf(fp,"    - %d GPUs (GPU ids: %d",info.n_gpu,info.p_gpu[0]);
        for(int i=1;i<info.n_gpu;i++)
            fprintf(fp,",%d",info.p_gpu[i]);
        fprintf(fp,"), ");
    }
    else {
        fprintf(fp,"    - 1 GPU (GPU id: %d), ",info.p_gpu[0]);
    }
    
    if( info.n_threads > 1 ) {
        fprintf(fp,"and %d threads.\n",info.n_threads);
    }
    else{
        fprintf(fp,"and 1 thread.\n");
    }
	
	fprintf(fp,"    - Bandpass: [%.1f - %.1f] ",info.fpix_min,info.fpix_max);
    if( info.fpix_roll > 0 )
        fprintf(fp," (roll off: %.2f).\n",info.fpix_roll);
    else
        fprintf(fp,".\n");

    fprintf(fp,"    - ");
    if( info.pad_size > 0 ) {
        if( info.pad_type == PAD_ZERO )
            fprintf(fp,"Zero padding. ");
        if( info.pad_type == PAD_GAUSSIAN )
            fprintf(fp,"Random padding. ");
    }

    if( info.ctf_type == NO_INV )
        fprintf(fp,"No CTF. ");
    if( info.ctf_type == PHASE_FLIP )
        fprintf(fp,"Phase-flip. ");
    if( info.ctf_type == WIENER_INV )
        fprintf(fp,"Wiener Inversion. ");
    if( info.ctf_type == WIENER_INV_SSNR )
        fprintf(fp,"Wiener SSNR: S=%.2f F=%.2f. ",info.ssnr_S,info.ssnr_F);
    
    if( info.norm_type == NO_NORM )
        fprintf(fp,"No Normalization.\n");
    if( info.norm_type == ZERO_MEAN )
        fprintf(fp,"Normalized to Mean=0.\n");
    if( info.norm_type == ZERO_MEAN_1_STD )
        fprintf(fp,"Normalized to Mean=0, Std=1.\n");
    if( info.norm_type == ZERO_MEAN_W_STD )
        fprintf(fp,"Normalized to Mean=0, Std=PRJ_W.\n");

    fprintf(fp,"    - %s Symmetry. Inversion: Ite=%d Std=%f\n",info.sym,info.w_inv_ite,info.w_inv_std);
}

void print(const Info&info,FILE*fp=stdout) {
	if( info.verbosity > 0 )
		print_full(info,fp);
	else
		print_minimal(info,fp);
}


}

#endif /// RECONSTRUCT_ARGS_H

