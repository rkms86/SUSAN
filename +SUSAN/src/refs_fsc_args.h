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

#ifndef REFS_FSC_ARGS_H
#define REFS_FSC_ARGS_H

#include <pthread.h>
#include <getopt.h>
#include "datatypes.h"
#include "gpu.h"
#include "angles_provider.h"
#include "math_cpu.h"
#include "points_provider.h"
#include "reference.h"

namespace ArgsRefsFsc {

typedef struct {
    uint32 gpu_ix;
    float  box_size;
    float  pix_size;
    float  rand_fpix;
    float  threshold;
    bool   save_svg;

    char   refs_file[SUSAN_FILENAME_LENGTH];
    char   out_dir  [SUSAN_FILENAME_LENGTH];
} Info;

bool validate(Info&info) {
    bool rslt = true;
    if( !IO::exists(info.refs_file) ) {
        fprintf(stderr,"References file %s does not exist.\n",info.refs_file);
        rslt = false;
    }
    else {
        References refs(info.refs_file);
        uint32 X,Y,Z;
        Mrc::read_size(X,Y,Z,refs[0].map);
        info.box_size = (float)X;
        if( info.pix_size<=0 )
            info.pix_size = Mrc::get_apix(refs[0].map);
        refs.check_size(X,true);
    }
    if( strlen(info.out_dir) == 0 ) {
        fprintf(stderr,"Output folder is missing.\n");
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
    info.gpu_ix    = 0;
    info.box_size  = 0;
    info.pix_size  = -1;
    info.rand_fpix = 10;
    info.threshold = 0.143;
    info.save_svg  = false;
    memset(info.refs_file,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.out_dir  ,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    sprintf(info.out_dir,"./");

    /// Parse inputs:
    enum {
        REFS_FILE,
        GPU_ID,
        OUT_DIR,
        RAN_FPIX,
        PIX_SIZE,
        THRESHOLD,
        SAVE_SVG
    };

    int c;
    static struct option long_options[] = {
        {"out_dir",   1, 0, OUT_DIR   },
        {"refs_file", 1, 0, REFS_FILE },
        {"gpu_id",    1, 0, GPU_ID    },
        {"rand_fpix", 1, 0, RAN_FPIX  },
        {"pix_size",  1, 0, PIX_SIZE  },
        {"threshold", 1, 0, THRESHOLD },
        {"save_svg",  1, 0, SAVE_SVG  },
        {0, 0, 0, 0}
    };
    
    while( (c=getopt_long_only(ac, av, "", long_options, 0)) >= 0 ) {
        switch(c) {
            case OUT_DIR:
                strcpy(info.out_dir,optarg);
                break;
            case REFS_FILE:
                strcpy(info.refs_file,optarg);
                break;
            case GPU_ID:
                info.gpu_ix = atoi(optarg);
                break;
            case RAN_FPIX:
                info.rand_fpix = atof(optarg);
                break;
            case PIX_SIZE:
                info.pix_size = atof(optarg);
                break;
            case THRESHOLD:
                info.threshold = atof(optarg);
                break;
            case SAVE_SVG :
                info.save_svg = atoi(optarg)>0;
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
    fprintf(stdout,"\tFourier Shell Correlation\n");

    fprintf(stdout,"\t\tReference file: %s.\n",info.refs_file);
    fprintf(stdout,"\t\tOutput folder:  %s\n",info.out_dir);

    fprintf(stdout,"\t\tVolume size: %.0fx%.0fx%.0f. (Voxel size: %.3f)\n",info.box_size,info.box_size,info.box_size,info.pix_size);
    
    fprintf(stdout,"\t\tUsing 1 GPU (GPU id: %d).\n",info.gpu_ix);

    fprintf(stdout,"\t\tRandomizing phases after %.1f fourier pixels.\n",info.rand_fpix);
}

}

#endif /// REFS_ALIGNER_ARGS_H

