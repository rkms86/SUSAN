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

#ifndef CTF_REFINER_ARGS_H
#define CTF_REFINER_ARGS_H

#include <pthread.h>
#include <getopt.h>
#include "datatypes.h"
#include "gpu.h"
#include "angles_provider.h"
#include "math_cpu.h"
#include "points_provider.h"
#include "reference.h"

namespace ArgsCtfRef {

typedef enum {
    PAD_ZERO=0,
    PAD_GAUSSIAN
} PaddingType_t;

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
    float  delta_def;
    float  delta_ang;
    bool   est_dose;
    bool   use_halves;

    char   refs_file[SUSAN_FILENAME_LENGTH];
    char   ptcls_out[SUSAN_FILENAME_LENGTH];
    char   ptcls_in [SUSAN_FILENAME_LENGTH];
    char   tomo_file[SUSAN_FILENAME_LENGTH];
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
    if( !IO::exists(info.refs_file) ) {
        fprintf(stderr,"References file %s does not exist.\n",info.refs_file);
        rslt = false;
    }
    else {
        References refs(info.refs_file);
        if( !refs.check_fields(info.use_halves) ) {
            exit(1);
        }
        if( !refs.check_size(info.box_size,info.use_halves) ) {
            exit(1);
        }
    }
    if( !IO::exists(info.tomo_file) ) {
        fprintf(stderr,"Tomos file %s does not exist.\n",info.tomo_file);
        rslt = false;
    }
    if( strlen(info.ptcls_out) == 0 ) {
        fprintf(stderr,"Particles output is missing.\n");
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
}

bool parse_args(Info&info,int ac,char** av) {
    /// Default values:
    info.n_gpu      = 0;
    info.n_threads  = 1;
    info.box_size   = 200;
    info.fpix_min   = 5;
    info.fpix_max   = 40;
    info.fpix_roll  = 4;
    info.pad_size   = 0;
    info.pad_type   = PAD_ZERO;
    info.delta_def  = 1000;
    info.delta_ang  = 20;
    info.est_dose   = false;
    info.use_halves = false;

    memset(info.p_gpu    ,0,SUSAN_MAX_N_GPU*sizeof(uint32));
    memset(info.refs_file,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.ptcls_out,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.ptcls_in ,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.tomo_file,0,SUSAN_FILENAME_LENGTH*sizeof(char));

    /// Parse inputs:
    enum {
        TOMOS_FILE,
        PTCLS_OUT,
        PTCLS_IN,
        REFS_FILE,
        N_THREADS,
        GPU_LIST,
        BOX_SIZE,
        PAD_SIZE,
        PAD_TYPE,
        BANDPASS,
        ROLLOFF_F,
        USE_HALVES,
        DEF_RANGE,
        ANG_RANGE,
        EST_DOSE
    };

    int c;
    static struct option long_options[] = {
        {"tomos_file", 1, 0, TOMOS_FILE},
        {"ptcls_in",   1, 0, PTCLS_IN  },
        {"ptcls_out",  1, 0, PTCLS_OUT },
        {"refs_file",  1, 0, REFS_FILE },
        {"n_threads",  1, 0, N_THREADS },
        {"gpu_list",   1, 0, GPU_LIST  },
        {"box_size",   1, 0, BOX_SIZE  },
        {"pad_size",   1, 0, PAD_SIZE  },
        {"pad_type",   1, 0, PAD_TYPE  },
        {"bandpass",   1, 0, BANDPASS  },
        {"rolloff_f",  1, 0, ROLLOFF_F },
        {"def_range",  1, 0, DEF_RANGE },
        {"ang_range",  1, 0, ANG_RANGE },
        {"est_dose",   1, 0, EST_DOSE  },
        {"use_halves", 1, 0, USE_HALVES},
        {0, 0, 0, 0}
    };
    
    single *tmp_single;
    uint32 *tmp_uint32;
    while( (c=getopt_long_only(ac, av, "", long_options, 0)) >= 0 ) {
        switch(c) {
            case TOMOS_FILE:
                strcpy(info.tomo_file,optarg);
                break;
            case PTCLS_OUT:
                strcpy(info.ptcls_out,optarg);
                break;
            case PTCLS_IN:
                strcpy(info.ptcls_in,optarg);
                break;
            case REFS_FILE:
                strcpy(info.refs_file,optarg);
                break;
            case BOX_SIZE:
                info.box_size = Math::make_even_up((float)atoi(optarg));
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
                info.pad_size = Math::make_even_up((float)atoi(optarg));
                break;
            case PAD_TYPE:
                info.pad_type = get_pad_type(optarg);
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
            case DEF_RANGE:
                info.delta_def = atof(optarg);
                break;
            case ANG_RANGE:
                info.delta_ang = atof(optarg);
                break;
            case EST_DOSE:
                info.est_dose = (atoi(optarg)>0);
                break;
            case USE_HALVES:
                info.use_halves = (atoi(optarg)>0);
                break;
            default:
                printf("Unknown parameter %d\n",c);
                exit(1);
                break;
        } /// switch
    } /// while(c)

    return validate(info);
}

void print(const Info&info,FILE*fp=stdout) {
    fprintf(stdout,"\tCTF Refiner");
    if( info.use_halves )
        fprintf(stdout," (independent half-sets)");
    fprintf(stdout,":\n");

    fprintf(stdout,"\t\tParticles in:   %s.\n",info.ptcls_in);
    fprintf(stdout,"\t\tTomograms file: %s.\n",info.tomo_file);
    fprintf(stdout,"\t\tReference file: %s.\n",info.refs_file);
    fprintf(stdout,"\t\tParticles out:  %s.\n",info.ptcls_out);

    fprintf(stdout,"\t\tVolume size: %dx%dx%d",info.box_size,info.box_size,info.box_size);
    if( info.pad_size > 0 ) {
        fprintf(stdout,", with padding of %d voxels",info.pad_size);
    }
    fprintf(stdout,".\n");
    
    if( info.n_gpu > 1 ) {
        fprintf(stdout,"\t\tUsing %d GPUs (GPU ids: %d",info.n_gpu,info.p_gpu[0]);
        for(int i=1;i<info.n_gpu;i++)
            fprintf(stdout,",%d",info.p_gpu[i]);
        fprintf(stdout,"), ");
    }
    else {
        fprintf(stdout,"\t\tUsing 1 GPU (GPU id: %d), ",info.p_gpu[0]);
    }
    
    if( info.n_threads > 1 ) {
        fprintf(stdout,"and %d threads.\n",info.n_threads);
    }
    else{
        fprintf(stdout,"and 1 thread.\n");
    }

    fprintf(stdout,"\t\tBandpass: [%.1f - %.1f] fourier pixels",info.fpix_min,info.fpix_max);
    if( info.fpix_roll > 0 )
        fprintf(stdout," with a roll off of %.2f.\n",info.fpix_roll);
    else
        fprintf(stdout,".\n");

    if( info.pad_size > 0 ) {
        if( info.pad_type == PAD_ZERO )
            fprintf(stdout,"\t\tPadding policy: Fill with zeros.\n");
        if( info.pad_type == PAD_GAUSSIAN )
            fprintf(stdout,"\t\tPadding policy: Fill with gaussian noise.\n");
    }

    fprintf(stdout,"\t\tDefocus search range: %.2f.\n",info.delta_def);
    fprintf(stdout,"\t\tDefocus angle range: %.2f.\n",info.delta_ang);

    if( info.est_dose )
        fprintf(stdout,"\t\tWith dose weighting estimation.\n");
    else
        fprintf(stdout,"\t\tWithout dose weighting estimation.\n");
}

}

#endif /// ALIGNER_ARGS_H

