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



