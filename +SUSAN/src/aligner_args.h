/*
 * This file is part of the Substack Analysis (SUSAN) framework.
 * Copyright (c) 2018-2021 Ricardo Miguel Sanchez Loayza.
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

#ifndef ALIGNER_ARGS_H
#define ALIGNER_ARGS_H

#include <pthread.h>
#include <getopt.h>
#include "datatypes.h"
#include "gpu.h"
#include "angles_provider.h"
#include "math_cpu.h"
#include "points_provider.h"
#include "reference.h"

namespace ArgsAli {

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
    ON_REFERENCE,
    ON_SUBSTACK,
    ON_SUBSTACK_SSNR,
    ON_SUBSTACK_WHITENING,
    ON_SUBSTACK_PHASE,
    CUMULATIVE_FSC
} CtfCorrectionType_t;

typedef enum {
    ELLIPSOID,
    CYLINDER,
    CIRCLE,
} OffsetType_t;

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
    uint32 type;
    float  ssnr_F;
    float  ssnr_S;
    bool   ali_halves;
    bool   drift;
    float  cone_range;
    float  cone_step;
    float  inplane_range;
    float  inplane_step;
    uint32 refine_level;
    uint32 refine_factor;
    uint32 off_type;
    float  off_x;
    float  off_y;
    float  off_z;
    float  off_s;

    char   pseudo_sym[64];

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
    uint32 rslt = ON_REFERENCE;
    bool all_ok = false;

    if( strcmp(arg,"none") == 0 ) {
        rslt = NO_INV;
        all_ok = true;
    }

    if( strcmp(arg,"on_reference") == 0 ) {
        rslt = ON_REFERENCE;
        all_ok = true;
    }

    if( strcmp(arg,"on_substack") == 0 ) {
        rslt = ON_SUBSTACK;
        all_ok = true;
    }

    if( strcmp(arg,"wiener_ssnr") == 0 ) {
        rslt = ON_SUBSTACK_SSNR;
        all_ok = true;
    }

    if( strcmp(arg,"wiener_white") == 0 ) {
        rslt = ON_SUBSTACK_WHITENING;
        all_ok = true;
    }

    if( strcmp(arg,"wiener_phase") == 0 ) {
        rslt = ON_SUBSTACK_PHASE;
        all_ok = true;
    }

    if( strcmp(arg,"cfsc") == 0 ) {
        rslt = CUMULATIVE_FSC;
        all_ok = true;
    }

    if( !all_ok ) {
        fprintf(stderr,"Invalid ctf correction type %s. Options are: none, on_reference, on_substack, wiener_ssnr and wiener_white. Defaulting to on_reference.\n",arg);
    }

    return rslt;
}

void set_symmetry(Info&info,const char*arg) {

    uint32 num_angs;
    M33f*p_angs;

    p_angs = AnglesSymmetry::get_rotation_list(num_angs,arg);

    if( num_angs == 0  ) {
        strcpy(info.pseudo_sym,"c1");
        fprintf(stderr,"Invalid symmetry option %s. Supported: c1, c2, cXXX... Defaulting to c1.\n",arg);
    }
    else {
        strcpy(info.pseudo_sym,arg);
        delete [] p_angs;
    }
}

uint32 get_offset_type(const char*arg) {
    uint32 rslt = ELLIPSOID;
    bool all_ok = false;

    if( strcmp(arg,"ellipsoid") == 0 ) {
        rslt = ELLIPSOID;
        all_ok = true;
    }

    if( strcmp(arg,"cylinder") == 0 ) {
        rslt = CYLINDER;
        all_ok = true;
    }

    if( !all_ok ) {
        fprintf(stderr,"Invalid offset type %s. Options are: ellipsoid and cylinder. Defaulting to ellipsoid.\n",arg);
    }

    return rslt;
}

bool validate(const Info&info) {
    bool rslt = true;
    if( info.type < 2 || info.type > 3 ) {
        fprintf(stderr,"Invalid type %d. Use 3 for 3D alignment or 2 for 2D.\n",info.type);
        rslt = false;
    }
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
        if( !refs.check_fields(info.ali_halves) ) {
            exit(1);
        }
        if( !refs.check_size(info.box_size,info.ali_halves) ) {
            exit(1);
        }
    }
    if( !IO::exists(info.tomo_file) ) {
        fprintf(stderr,"Tomos file %s does not exist.\n",info.tomo_file);
        rslt = false;
    }
    if( strlen(info.ptcls_in) == 0 ) {
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
    info.n_gpu         = 0;
    info.n_threads     = 1;
    info.box_size      = 200;
    info.fpix_min      = 0;
    info.fpix_max      = 30;
    info.fpix_roll     = 4;
    info.pad_size      = 0;
    info.pad_type      = PAD_ZERO;
    info.ctf_type      = ON_REFERENCE;
    info.norm_type     = NO_NORM;
    info.type          = 3;
    info.ssnr_F        = 0;
    info.ssnr_S        = 1;
    info.ali_halves    = false;
    info.drift         = true;
    info.cone_range    = 0;
    info.cone_step     = 1;
    info.inplane_range = 0;
    info.inplane_step  = 1;
    info.refine_level  = 0;
    info.refine_factor = 1;
    info.off_type      = ELLIPSOID;
    info.off_x         = 0;
    info.off_y         = 0;
    info.off_z         = 0;
    info.off_s         = 1;
    memset(info.p_gpu    ,0,SUSAN_MAX_N_GPU*sizeof(uint32));
    memset(info.refs_file,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.ptcls_out,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.ptcls_in ,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.tomo_file,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    strcpy(info.pseudo_sym,"c1");

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
        NORM_TYPE,
        CTF_TYPE,
        SSNR,
        BANDPASS,
        ROLLOFF_F,
        SYMMETRY,
        ALI_HALVES,
        DRIFT,
        CONE,
        INPLANE,
        REFINE,
        OFF_TYPE,
        OFF_PARAM,
        TYPE
    };

    int c;
    static struct option long_options[] = {
        {"tomos_file",  1, 0, TOMOS_FILE},
        {"ptcls_in",    1, 0, PTCLS_IN  },
        {"ptcls_out",   1, 0, PTCLS_OUT },
        {"refs_file",   1, 0, REFS_FILE },
        {"n_threads",   1, 0, N_THREADS },
        {"gpu_list",    1, 0, GPU_LIST  },
        {"box_size",    1, 0, BOX_SIZE  },
        {"pad_size",    1, 0, PAD_SIZE  },
        {"pad_type",    1, 0, PAD_TYPE  },
        {"norm_type",   1, 0, NORM_TYPE },
        {"ctf_type",    1, 0, CTF_TYPE  },
        {"ssnr_param",  1, 0, SSNR      },
        {"bandpass",    1, 0, BANDPASS  },
        {"rolloff_f",   1, 0, ROLLOFF_F },
        {"p_symmetry",  1, 0, SYMMETRY  },
        {"ali_halves",  1, 0, ALI_HALVES},
        {"allow_drift", 1, 0, DRIFT     },
        {"cone",        1, 0, CONE      },
        {"inplane",     1, 0, INPLANE   },
        {"refine",      1, 0, REFINE    },
        {"off_type",    1, 0, OFF_TYPE  },
        {"off_params",  1, 0, OFF_PARAM },
        {"type",        1, 0, TYPE },
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
            case SYMMETRY:
                set_symmetry(info,optarg);
                break;
            case ALI_HALVES:
                info.ali_halves = (atoi(optarg)>0);
                break;
            case DRIFT:
                info.drift = (atoi(optarg)>0);
                break;
            case CONE:
                IO::parse_single_strlist(tmp_single, optarg);
                info.cone_range = tmp_single[0];
                info.cone_step  = tmp_single[1];
                delete [] tmp_single;
                break;
            case INPLANE:
                IO::parse_single_strlist(tmp_single, optarg);
                info.inplane_range = tmp_single[0];
                info.inplane_step  = tmp_single[1];
                delete [] tmp_single;
                break;
            case REFINE:
                IO::parse_uint32_strlist(tmp_uint32, optarg);
                info.refine_factor  = tmp_uint32[0];
                info.refine_level = tmp_uint32[1];
                delete [] tmp_uint32;
                break;
            case OFF_TYPE:
                info.off_type = get_offset_type(optarg);
                break;
            case OFF_PARAM:
                IO::parse_single_strlist(tmp_single, optarg);
                info.off_x = tmp_single[0];
                info.off_y = tmp_single[1];
                info.off_z = tmp_single[2];
                info.off_s = tmp_single[3];
                delete [] tmp_single;
                break;
            case TYPE:
                info.type = atoi(optarg);
                break;
            default:
                printf("Unknown parameter %d\n",c);
                exit(1);
                break;
        } /// switch
    } /// while(c)
    
    if( info.type == 2 ) {
        info.off_type = CIRCLE;
        info.off_s = 1;
    }

    return validate(info);
}

void print_angles(const Info&info) {
	
    AnglesProvider angles;
    angles.cone_range    = info.cone_range;
    angles.cone_step     = info.cone_step;
    angles.inplane_range = info.inplane_range;
    angles.inplane_step  = info.inplane_step;
    angles.refine_level  = info.refine_level;
    angles.refine_factor = info.refine_factor;
    angles.set_symmetry(info.pseudo_sym);

    int count_total=0;
    int count_level=0;
    int lvl = 0;

    for( angles.levels_init(); angles.levels_available(); angles.levels_next(), lvl++ ) {
        count_level=0;
        for( angles.sym_init(); angles.sym_available(); angles.sym_next() ) {
            for( angles.cone_init(); angles.cone_available(); angles.cone_next() ) {
                for( angles.inplane_init(); angles.inplane_available(); angles.inplane_next() ) {
                    count_level++;
                } /// inplane
            } /// cone
        } /// symmetry
        fprintf(stdout,"\t\tAngles in refinement level %2d: %7d.\n",lvl,count_level);
        count_total += count_level;
    } /// level
    fprintf(stdout,"\t\tTotal angles: %d.\n",count_total);
}

void print(const Info&info,FILE*fp=stdout) {
    fprintf(stdout,"\tVolume %dD alignment",info.type);
    if( info.ali_halves )
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
		
    if( info.ctf_type == NO_INV )
        fprintf(stdout,"\t\tCTF correction policy: Disabled.\n");
    if( info.ctf_type == ON_REFERENCE )
        fprintf(stdout,"\t\tCTF correction policy: On reference.\n");
    if( info.ctf_type == ON_SUBSTACK )
        fprintf(stdout,"\t\tCTF correction policy: On substack - Wiener inversion.\n");
    if( info.ctf_type == ON_SUBSTACK_SSNR )
        fprintf(stdout,"\t\tCTF correction policy: On substack - Wiener inversion with SSNR(f) = (100^(3*%.2f))*e^(-100*%.2f*f).\n",info.ssnr_S,info.ssnr_F);
    if( info.ctf_type == ON_SUBSTACK_WHITENING )
        fprintf(stdout,"\t\tCTF correction policy: On substack - Wiener inversion with whitening filter (experimental).\n");
    if( info.ctf_type == ON_SUBSTACK_PHASE )
        fprintf(stdout,"\t\tCTF correction policy: On substack - Wiener inversion with phase cross-correlation (experimental).\n");
    if( info.ctf_type == CUMULATIVE_FSC )
        fprintf(stdout,"\t\tCTF correction policy: On substack - Wiener inversion with cumulative FSC (experimental).\n");

    if( info.norm_type == NO_NORM )
        fprintf(stdout,"\t\tSubstack normalization policy: Disabled.\n");
    if( info.norm_type == ZERO_MEAN )
        fprintf(stdout,"\t\tSubstack normalization policy: Mean=0.\n");
    if( info.norm_type == ZERO_MEAN_1_STD )
        fprintf(stdout,"\t\tSubstack normalization policy: Mean=0, Std=1.\n");
    if( info.norm_type == ZERO_MEAN_W_STD )
        fprintf(stdout,"\t\tSubstack normalization policy: Mean=0, Std according to projection weight.\n");

    fprintf(stdout,"\t\tPseudo-symmetry search: %s.\n",info.pseudo_sym);
    fprintf(stdout,"\t\tCone search:    Range=%.3f, Step=%.3f.\n",info.cone_range,info.cone_step);
    fprintf(stdout,"\t\tInplane search: Range=%.3f, Step=%.3f.\n",info.inplane_range,info.inplane_step);
    fprintf(stdout,"\t\tAngle refinement: Levels=%d, Factor=%d.\n",info.refine_level,info.refine_factor);
    print_angles(info);
	
    uint32_t total_points=0;
    if( info.off_type == ELLIPSOID ) {
        Vec3*pt = PointsProvider::ellipsoid(total_points,info.off_x,info.off_y,info.off_z,info.off_s);
        fprintf(stdout,"\t\tEllipsoid offset search (3D): ");
        delete [] pt;
    }
    if( info.off_type == CYLINDER ) {
        Vec3*pt = PointsProvider::cylinder(total_points,info.off_x,info.off_y,info.off_z,info.off_s);
        fprintf(stdout,"\t\tCylindrical offset search (3D): ");
        delete [] pt;
    }
    if( info.off_type == CIRCLE ) {
        Vec3*pt = PointsProvider::cylinder(total_points,info.off_x,info.off_y,info.off_z,info.off_s);
        fprintf(stdout,"\t\tCircular offset search (2D): ");
        delete [] pt;
    }
	
    fprintf(stdout,"Range=[%.2f,%.2f,%.2f], Step=%.2f. Total points: %d\n",info.off_x,info.off_y,info.off_z,info.off_s,total_points);
}

}

#endif /// ALIGNER_ARGS_H

