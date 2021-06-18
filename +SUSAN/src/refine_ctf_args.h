#ifndef REFINE_CTF_ARGS_H
#define REFINE_CTF_ARGS_H

#include <pthread.h>
#include <getopt.h>
#include "datatypes.h"
#include "gpu.h"
#include "angles_provider.h"
#include "math_cpu.h"
#include "points_provider.h"
#include "reference.h"

namespace RefCtfAli {

typedef struct {
    int    n_gpu;
    uint32 p_gpu[SUSAN_MAX_N_GPU];
    uint32 n_threads;
    uint32 box_size;
    single res_min;
    single res_max;
    single def_min;
    single def_max;
    single astg_ang;
    single astg_def;
    int    verbose_id;
    char   ptcls_out[SUSAN_FILENAME_LENGTH];
    char   ptcls_in [SUSAN_FILENAME_LENGTH];
    char   tomo_file[SUSAN_FILENAME_LENGTH];
} Info;

bool validate(const Info&info) {
    bool rslt = true;
    if( info.res_min <= info.res_max ) {
            fprintf(stderr,"Invalid resolution range: %f - %f.\n",info.res_min,info.res_max);
            rslt = false;
    }
    if( info.def_max <= info.def_min ) {
            fprintf(stderr,"Invalid defocus range: %f - %f.\n",info.def_min,info.def_max);
            rslt = false;
    }
    if( !IO::exists(info.ptcls_in) ) {
        fprintf(stderr,"Particles file %s does not exist.\n",info.ptcls_in);
        rslt = false;
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
    info.n_gpu      = 0;
    info.n_threads  = 1;
    info.box_size   = 324;
    info.res_min    = 0;
    info.res_max    = 0;
    info.def_min    = 0;
    info.def_max    = 0;
    info.astg_ang   = 10;
    info.astg_def   = 2000;
    info.verbose_id = -1;
    memset(info.p_gpu    ,0,SUSAN_MAX_N_GPU*sizeof(uint32));
    memset(info.ptcls_out,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.ptcls_in ,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.tomo_file,0,SUSAN_FILENAME_LENGTH*sizeof(char));

    /// Parse inputs:
    enum {
        TOMOS_FILE,
        PTCLS_OUT,
        PTCLS_IN,
        N_THREADS,
        GPU_LIST,
        BOX_SIZE,
        RES_RANGE,
        DEF_RANGE,
        ASTG_RANGE,
        VERBOSE_ID
    };

    int c;
    static struct option long_options[] = {
        {"tomos_file",  1, 0, TOMOS_FILE},
        {"ptcls_in",    1, 0, PTCLS_IN  },
        {"ptcls_out",   1, 0, PTCLS_OUT },
        {"n_threads",   1, 0, N_THREADS },
        {"gpu_list",    1, 0, GPU_LIST  },
        {"box_size",    1, 0, BOX_SIZE  },
        {"res_range",   1, 0, RES_RANGE },
        {"def_range",   1, 0, DEF_RANGE },
        {"astg_range",  1, 0, ASTG_RANGE},
        {"verbose_id",  1, 0, VERBOSE_ID},
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
            case RES_RANGE:
                IO::parse_single_strlist(tmp_single, optarg);
                info.res_min = tmp_single[0];
                info.res_max = tmp_single[1];
                delete [] tmp_single;
                break;
            case DEF_RANGE:
                IO::parse_single_strlist(tmp_single, optarg);
                info.def_min = tmp_single[0];
                info.def_max = tmp_single[1];
                delete [] tmp_single;
                break;
            case ASTG_RANGE:
                IO::parse_single_strlist(tmp_single, optarg);
                info.astg_ang = tmp_single[0];
                info.astg_def = tmp_single[1];
                delete [] tmp_single;
                break;
            case VERBOSE_ID:
                info.verbose_id = atoi(optarg);
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
    fprintf(fp,"\tCTF per particle per projection refinement\n");

    fprintf(fp,"\t\tParticles in:   %s.\n",info.ptcls_in);
    fprintf(fp,"\t\tParticles out:  %s.\n",info.ptcls_out);
    fprintf(fp,"\t\tTomograms file: %s.\n",info.tomo_file);

    fprintf(stdout,"\t\tPatch size: %dx%d\n",info.box_size,info.box_size);
    
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

    fprintf(stdout,"\t\tDefocus range: %.1f - %.1f angstroms.\n",info.res_min,info.res_max);
    fprintf(stdout,"\t\tResolution range: %.2f - %.2f angstroms.\n",info.def_min,info.def_max);
    fprintf(stdout,"\t\tAstigmatism search: +-%.2f degrees, +-%.2f angstroms.\n",info.astg_ang,info.astg_def);

    if( info.verbose_id >= 0 ) {
        fprintf(fp,"\t\tVerbose on particle id: %d\n",info.verbose_id);
    }
}

}

#endif /// REFINE_CTF_ARGS_H

