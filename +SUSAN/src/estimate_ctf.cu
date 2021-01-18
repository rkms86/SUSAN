#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <unistd.h>

#include "io.h"
#include "args_parse.h"
#include "estimate_ctf.h"

#include "mrc.h"

int main(int ac, char** av) {

	ArgsCTF::Info info;

	if( ArgsCTF::parse_args(info,ac,av) ) {
		ArgsCTF::print(info);
		PBarrier barrier(2);
		ParticlesRW ptcls(info.ptcls_in);
		Tomograms tomos(info.tomos_in);
		StackReader stkrdr(&ptcls,&tomos,&barrier);
		CtfEstimatePool pool(&info,tomos.num_proj,stkrdr,info.n_threads);
		
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



