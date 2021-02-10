#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <unistd.h>

#include "io.h"
#include "mrc.h"
#include "rec_tomograms.h"
#include "particles.h"
#include "tomogram.h"
#include "tomo_generator.h"
#include "rec_tomograms_args.h"

#include "math_cpu.h"

void print_data_info(Tomograms&tomos) {
	printf("\t\tTomograms available:  %d.\n",tomos.num_tomo);
	printf("\t\tAvailabe projections: %d (max).\n",tomos.num_proj);
}

int main(int ac, char** av) {

	ArgsRecTomos::Info info;

	if( ArgsRecTomos::parse_args(info,ac,av) ) {
		ArgsRecTomos::print(info);
		Tomograms tomos(info.tomos_in);
		TomoParticles ptcls(info.box_size,tomos);
		print_data_info(tomos);
		PBarrier barrier(2);
		StackReader stkrdr(&ptcls,&tomos,&barrier);
		RecTomogramsPool pool(&info,tomos.num_proj,stkrdr,info.n_threads);
				
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



