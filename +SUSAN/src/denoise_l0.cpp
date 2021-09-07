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
#include "mrc.h"
#include "math_cpu.h"
#include "denoise_l0.h"

int main(int ac, char** av) {

    DenL0Rec::Info info;

    if( DenL0Rec::parse_args(info,ac,av) ) {
        DenL0Rec::print(info);


        if( !Mrc::is_mode_float(info.map_name) ) {
            fprintf(stderr,"Map %s must be mode 2.\n",info.map_name);
            exit(1);
        }

        if( info.rho > 0 ) {
            float apix = Mrc::get_apix(info.map_name);
            uint32 X,Y,Z;
            float *data = Mrc::read(X,Y,Z,info.map_name);

            /// create back-up
            char bak_name[SUSAN_FILENAME_LENGTH];
            strcpy(bak_name,info.map_name);
            strcpy(bak_name+strlen(info.map_name)-4,"_raw.mrc");
            Mrc::write(data,X,Y,Z,bak_name);
            Mrc::set_apix(bak_name,apix,X,Y,Z);

            /// denoise:
            float *new_data = new float[X*Y*Z];
            Math::denoise_l0(new_data,data,X*Y*Z,info.lambda,info.rho);
            Mrc::write(new_data,X,Y,Z,info.map_name);
            Mrc::set_apix(info.map_name,apix,X,Y,Z);

            delete [] data;
            delete [] new_data;
        }

    }
    else {
        fprintf(stderr,"Error parsing input arguments.\n");
        exit(1);
    }

    return 0;
}



