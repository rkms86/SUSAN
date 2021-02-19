#ifndef REFS_ALIGNER_ARGS_H
#define REFS_ALIGNER_ARGS_H

#include <pthread.h>
#include <getopt.h>
#include "datatypes.h"
#include "gpu.h"
#include "angles_provider.h"
#include "math_cpu.h"
#include "points_provider.h"
#include "reference.h"

namespace ArgsRefsAli {

typedef enum {
    ELLIPSOID,
    CYLINDER,
    CIRCLE,
} OffsetType_t;

typedef struct {
    uint32 gpu_ix;
    uint32 box_size;
    single fpix_min;
    single fpix_max;
    single fpix_roll;
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

    char   refs_file[SUSAN_FILENAME_LENGTH];
    char   ptcls_out[SUSAN_FILENAME_LENGTH];
    char   ptcls_in [SUSAN_FILENAME_LENGTH];
} Info;

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
        refs.check_size(info.box_size,true);
    }
    if( strlen(info.ptcls_in) == 0 ) {
        fprintf(stderr,"Particles output is missing.\n");
        rslt = false;
    }

    int available_gpus = GPU::count_devices();
    if(available_gpus==0) {
        fprintf(stderr,"Not available GPUs on the system.\n");
        rslt = false;
    }
    else {
        if( info.gpu_ix >= available_gpus ) {
            fprintf(stderr,"Requesting unavalable GPU with ID %d.\n",info.gpu_ix);
            rslt = false;
        }
    }

    return rslt;
}

bool parse_args(Info&info,int ac,char** av) {
    /// Default values:
    info.gpu_ix        = 0;
    info.box_size      = 200;
    info.fpix_min      = 0;
    info.fpix_max      = 30;
    info.fpix_roll     = 4;
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
    memset(info.refs_file,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.ptcls_out,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.ptcls_in ,0,SUSAN_FILENAME_LENGTH*sizeof(char));

    /// Parse inputs:
    enum {
        PTCLS_OUT,
        PTCLS_IN,
        REFS_FILE,
        GPU_ID,
        BOX_SIZE,
        BANDPASS,
        ROLLOFF_F,
        CONE,
        INPLANE,
        REFINE,
        OFF_TYPE,
        OFF_PARAM
    };

    int c;
    static struct option long_options[] = {
        {"ptcls_in",    1, 0, PTCLS_IN  },
        {"ptcls_out",   1, 0, PTCLS_OUT },
        {"refs_file",   1, 0, REFS_FILE },
        {"gpu_id",      1, 0, GPU_ID    },
        {"box_size",    1, 0, BOX_SIZE  },
        {"bandpass",    1, 0, BANDPASS  },
        {"rolloff_f",   1, 0, ROLLOFF_F },
        {"cone",        1, 0, CONE      },
        {"inplane",     1, 0, INPLANE   },
        {"refine",      1, 0, REFINE    },
        {"off_type",    1, 0, OFF_TYPE  },
        {"off_params",  1, 0, OFF_PARAM },
        {0, 0, 0, 0}
    };
    
    single *tmp_single;
    uint32 *tmp_uint32;
    while( (c=getopt_long_only(ac, av, "", long_options, 0)) >= 0 ) {
        switch(c) {
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
            case GPU_ID:
                info.gpu_ix = atoi(optarg);
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
            default:
                printf("Unknown parameter %d\n",c);
                exit(1);
                break;
        } /// switch
    } /// while(c)

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
    fprintf(stdout,"\tHalf-set volumes alignment\n");

    fprintf(stdout,"\t\tParticles in:   %s.\n",info.ptcls_in);
    fprintf(stdout,"\t\tReference file: %s.\n",info.refs_file);
    fprintf(stdout,"\t\tParticles out:  %s.\n",info.ptcls_out);

    fprintf(stdout,"\t\tVolume size: %dx%dx%d.\n",info.box_size,info.box_size,info.box_size);
    
    fprintf(stdout,"\t\tUsing 1 GPU (GPU id: %d).\n",info.gpu_ix);

    fprintf(stdout,"\t\tBandpass: [%.1f - %.1f] fourier pixels",info.fpix_min,info.fpix_max);
    if( info.fpix_roll > 0 )
        fprintf(stdout," with a roll off of %.2f.\n",info.fpix_roll);
    else
        fprintf(stdout,".\n");

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
	
    fprintf(stdout,"Range=[%.2f,%.2f,%.2f], Step=%.2f. Total points: %d\n",info.off_x,info.off_y,info.off_z,info.off_s,total_points);
}

}

#endif /// REFS_ALIGNER_ARGS_H

