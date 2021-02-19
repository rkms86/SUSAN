#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <unistd.h>

#include "io.h"
#include "refs_aligner.h"
#include "particles.h"
#include "tomogram.h"
#include "reference.h"
#include "refs_aligner_args.h"

void print_data_info(Particles&ptcls) {
	printf("\t\tAvailable particles:  %d.\n",ptcls.n_ptcl);
        printf("\t\tNumber of classes:    %d.\n",ptcls.n_refs);
}

int main(int ac, char** av) {

        ArgsRefsAli::Info info;

        if( ArgsRefsAli::parse_args(info,ac,av) ) {
                ArgsRefsAli::print(info);
                ParticlesRW ptcls(info.ptcls_in);
		References refs(info.refs_file);
                print_data_info(ptcls);

                RefsAligner aligner(&info,&refs);
                aligner.align();

                //ptcls.save(info.ptcls_out);
	}
	else {
		fprintf(stderr,"Error parsing input arguments.\n");
		exit(1);
	}
	
    return 0;
}



