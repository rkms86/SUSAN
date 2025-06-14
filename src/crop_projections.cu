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
#include "crop_projections.h"
#include "particles.h"
#include "tomogram.h"
#include "crop_projections_args.h"

void print_data_info(Particles&ptcls,Tomograms&tomos) {
    printf("\t\tAvailable particles:  %d.\n",ptcls.n_ptcl);
    printf("\t\tNumber of classes:    %d.\n",ptcls.n_refs);
    printf("\t\tTomograms available:  %d.\n",tomos.num_tomo);
    printf("\t\tAvailabe projections: %d (max).\n",tomos.num_proj);
}

int main(int ac, char** av) {

    ArgsCropProjections::Info info;

    if( ArgsCropProjections::parse_args(info,ac,av) ) {
        ArgsCropProjections::print(info);
        PBarrier barrier(2);
        ParticlesRW ptcls(info.ptcls_in);
        Tomograms tomos(info.tomos_in);
        print_data_info(ptcls,tomos);
        StackReader stkrdr(&ptcls,&tomos,&barrier);
        CropProjectionsPool pool(&info,tomos.num_proj,ptcls.n_ptcl,stkrdr,info.n_threads);

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



