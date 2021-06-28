#ifndef DENOISE_L0_ARGS_H
#define DENOISE_L0_ARGS_H

#include <pthread.h>
#include <getopt.h>
#include "datatypes.h"

namespace DenL0Rec {

typedef struct {
    float  lambda;
    float  rho;
    char   map_name[SUSAN_FILENAME_LENGTH];
} Info;

bool validate(Info&info) {
    bool rslt = true;
    if( info.rho < 0 ) {
        fprintf(stderr,"Rho value %f invalid, setting it to 0.\n",info.rho);
        info.rho = 0;
    }

    if( info.rho > 1 ) {
        fprintf(stderr,"Rho value %f invalid, setting it to 1.\n",info.rho);
        info.rho = 1;
    }

    if( !IO::exists(info.map_name) ) {
        fprintf(stderr,"Map file %s does not exist.\n",info.map_name);
        rslt = false;
    }
    return rslt;
};

bool parse_args(Info&info,int ac,char** av) {
    /// Default values:
    info.lambda = -1;
    info.rho    = 0.5;
    memset(info.map_name ,0,SUSAN_FILENAME_LENGTH*sizeof(char));
	
	/// Parse inputs:
    enum {
        MAP_FILE,
        LAMBDA,
        RHO
    };

    int c;
    static struct option long_options[] = {
        {"map_file",  1, 0, MAP_FILE},
        {"lambda",    1, 0, LAMBDA},
        {"rho",       1, 0, RHO},
        {0, 0, 0, 0}
    };
    while( (c=getopt_long_only(ac, av, "", long_options, 0)) >= 0 ) {
        switch(c) {
            case MAP_FILE:
                strcpy(info.map_name,optarg);
                break;
            case LAMBDA:
                info.lambda = atof(optarg);
                break;
            case RHO:
                info.rho = atof(optarg);
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
    fprintf(stdout,"\tVolume l0-denoising:\n");
    fprintf(stdout,"\t\tVolume file: %s.\n",info.map_name);
    fprintf(stdout,"\t\tLambda: %f, Rho: %f.\n",info.lambda,info.rho);
}


}

#endif /// RECONSTRUCT_ARGS_H

