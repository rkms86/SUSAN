#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <unistd.h>

#include "io.h"
#include "reconstruct.h"
#include "particles.h"
#include "tomogram.h"
#include "reconstruct_args.h"

void print_data_info(Particles&ptcls,Tomograms&tomos) {
        printf("\t\tAvailable particles:  %d.\n",ptcls.n_ptcl);
        printf("\t\tNumber of classes:    %d.\n",ptcls.n_refs);
	printf("\t\tTomograms available:  %d.\n",tomos.num_tomo);
	printf("\t\tAvailabe projections: %d (max).\n",tomos.num_proj);
}

int main(int ac, char** av) {

    ArgsRec::Info info;

    if( ArgsRec::parse_args(info,ac,av) ) {

        ArgsRec::print(info);

        PBarrier barrier(2);

        printf("\tLoading data files..."); fflush(stdout);
        ParticlesRW ptcls(info.ptcls_in);
        Tomograms tomos(info.tomos_in);
        printf(" Done\n"); fflush(stdout);

        print_data_info(ptcls,tomos);

        StackReader stkrdr(&ptcls,&tomos,&barrier);
        RecPool pool(&info,ptcls.n_refs,tomos.num_proj,ptcls.n_ptcl,stkrdr,info.n_threads);

        stkrdr.start();
        pool.start();

        stkrdr.wait();
        pool.wait();
    }
    else {
        fprintf(stderr,"Error parsing input arguments.\n");
        exit(1);
    }

    return 0;
}



