#ifndef ESTIMATE_CTF_PARSE_H
#define ESTIMATE_CTF_PARSE_H

#include <pthread.h>
#include <getopt.h>
#include "datatypes.h"
#include "gpu.h"

namespace ArgsCTF {

typedef struct {
    uint32 n_threads;
    uint32 box_size;
    uint32 binning;
    single res_min;
    single res_max;
    single def_min;
    single def_max;
    single tlt_range;
    single ref_range;
    single ref_step;
    single res_thres;
    single bfac_max;
    int    verbose;
    int    n_gpu;
    uint32 p_gpu[SUSAN_MAX_N_GPU];
    char   out_dir[SUSAN_FILENAME_LENGTH];
    char   ptcls_in[SUSAN_FILENAME_LENGTH];
    char   tomos_in[SUSAN_FILENAME_LENGTH];
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
	if( !IO::exists(info.tomos_in) ) {
		fprintf(stderr,"Tomos file %s does not exist.\n",info.tomos_in);
		rslt = false;
	}
	IO::create_dir(info.out_dir);
	if( !IO::exist_dir(info.out_dir) ) {
		fprintf(stderr,"Output folder %s cannot be created.\n",info.out_dir);
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
	info.n_threads = 1;
	info.box_size  = 512;
	info.binning   = 0;
        info.res_thres = 0.5;
	info.res_min   = 0;
	info.res_max   = 0;
	info.def_min   = 0;
	info.def_max   = 0;
	info.tlt_range = 2000;
	info.ref_range = 2000;
	info.ref_step  = 100;
	info.bfac_max  = 700;
	info.n_gpu     = 0;
	info.verbose   = 0;
	memset(info.p_gpu   ,0,SUSAN_MAX_N_GPU*sizeof(uint32));
	memset(info.out_dir ,0,SUSAN_FILENAME_LENGTH*sizeof(char));
	memset(info.ptcls_in,0,SUSAN_FILENAME_LENGTH*sizeof(char));
	memset(info.tomos_in,0,SUSAN_FILENAME_LENGTH*sizeof(char));
	
	/// Parse inputs:
	enum {
        TOMOS_IN,
        DATA_OUT,
        RES_RANGE,
        RES_THRES,
        DEF_RANGE,
        TLT_RANGE,
        REFINE,
        BINNING,
        PTCLS_FILE,
        BOX_SIZE,
        N_THREADS,
        GPU_LIST,
        BFAC_MAX,
        VERBOSE
    };

    int c;
    static struct option long_options[] = {
        {"tomos_in",   1, 0, TOMOS_IN  },
        {"data_out",   1, 0, DATA_OUT  },
        {"ptcls_file", 1, 0, PTCLS_FILE},
        {"box_size",   1, 0, BOX_SIZE  },
        {"n_threads",  1, 0, N_THREADS },
        {"res_range",  1, 0, RES_RANGE },
        {"res_thres",  1, 0, RES_THRES },
        {"def_range",  1, 0, DEF_RANGE },
        {"tilt_search",1, 0, TLT_RANGE },
        {"refine_def" ,1, 0, REFINE    },
        {"binning",    1, 0, BINNING   },
        {"gpu_list",   1, 0, GPU_LIST  },
        {"verbose",    1, 0, VERBOSE   },
        {"bfactor_max",1, 0, BFAC_MAX  },
        {0, 0, 0, 0}
    };
    
    single *tmp_single;
    uint32 *tmp_uint32;
    float  tmp;
    while( (c=getopt_long_only(ac, av, "", long_options, 0)) >= 0 ) {
        switch(c) {
            case TOMOS_IN:
                strcpy(info.tomos_in,optarg);
                break;
            case DATA_OUT:
                strcpy(info.out_dir,optarg);
                break;
            case PTCLS_FILE:
                strcpy(info.ptcls_in,optarg);
                break;
            case BOX_SIZE:
                info.box_size = atoi(optarg);
                tmp = (float)(info.box_size);
                info.box_size = (int)(4.0*roundf(tmp/4)); // Force box to be multiple of 4.
                break;
            case N_THREADS:
                info.n_threads = atoi(optarg);
                break;
            case TLT_RANGE:
                info.tlt_range = atof(optarg);
                break;
            case BINNING:
                info.binning = atoi(optarg);
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
            case RES_THRES:
                info.res_thres = atof(optarg);
                break;
            case DEF_RANGE:
                IO::parse_single_strlist(tmp_single, optarg);
                info.def_min = tmp_single[0];
                info.def_max = tmp_single[1];
                delete [] tmp_single;
                break;
            case REFINE:
                IO::parse_single_strlist(tmp_single, optarg);
                info.ref_range = tmp_single[0];
                info.ref_step  = tmp_single[1];
                delete [] tmp_single;
                break;
            case BFAC_MAX:
                info.bfac_max = atof(optarg);
                break;
            case VERBOSE:
                info.verbose = atoi(optarg);
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
    fprintf(stdout,"\tCtf Estimation:\n");

    fprintf(stdout,"\t\tParticles file: %s.\n",info.ptcls_in);
    fprintf(stdout,"\t\tTomograms file: %s.\n",info.tomos_in);
    fprintf(stdout,"\t\tOutput folder: %s.\n",info.out_dir);

    fprintf(stdout,"\t\tPatch size: %dx%d, ",info.box_size,info.box_size);
    if( info.binning > 0 ) {
    fprintf(stdout,"bin level %d.\n",info.binning);
        }
    else {
        fprintf(stdout,"no binning.\n");
    }

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

    fprintf(stdout,"\t\tDefocus range: %.1f - %.1f angstroms.\n",info.res_min,info.res_max);
    fprintf(stdout,"\t\tResolution range: %.2f - %.2f angstroms.\n",info.def_min,info.def_max);
    fprintf(stdout,"\t\tTilt search range: %.1f angstroms.\n",info.tlt_range);
    fprintf(stdout,"\t\tDefocus refinement range: %.2f angstroms\n",info.ref_range);
    fprintf(stdout,"\t\tDefocus refinement step: %.2f angstroms\n",info.ref_step);
    
}

}

#endif /// ESTIMATE_CTF_PARSE_H

