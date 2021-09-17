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



