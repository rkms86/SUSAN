#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <unistd.h>

#include "io.h"
//#include "refs_aligner.h"
#include "particles.h"
#include "tomogram.h"
#include "reference.h"
#include "refs_fsc.h"
#include "refs_fsc_args.h"

void print_data_info(References&refs) {
    printf("\t\tNumber of classes:    %d.\n",refs.num_refs);
}

int main(int ac, char** av) {

    ArgsRefsFsc::Info info;

    if( ArgsRefsFsc::parse_args(info,ac,av) ) {
        ArgsRefsFsc::print(info);
        References refs(info.refs_file);
        print_data_info(refs);

        RefsFsc fsc(info);
        fsc.calc_fsc(refs);
        //RefsAligner aligner(&info,&refs);
        //aligner.align();
        //aligner.update_ptcls(ptcls);
        //ptcls.save(info.ptcls_out);
    }
    else {
        fprintf(stderr,"Error parsing input arguments.\n");
        exit(1);
    }

    return 0;
}



