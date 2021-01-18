#ifndef TOMOGRAM_H
#define TOMOGRAM_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "datatypes.h"
#include "memory.h"
#include "io.h"
#include "math_cpu.h"
#include "mrc.h"

class Tomogram {
public:
    uint32  tomo_id;
    VUInt3  tomo_dim;
    V3f     tomo_center;
    char    stk_name[ SUSAN_FILENAME_LENGTH ];
    VUInt3  stk_dim;
    V3f     stk_center;
    uint32  num_proj;
    single  pix_size;
    single  KV;
    single  CS;
    single  AC;
    M33f    *R;
    V3f     *t;
    single  *w;
    Defocus *def;

public:
    Tomogram() {
        tomo_id       = 0;
        tomo_dim.x    = 0;
        tomo_dim.y    = 0;
        tomo_dim.z    = 0;
        stk_name[0]   = 0;
        stk_dim.x     = 0;
        stk_dim.y     = 0;
        stk_dim.z     = 0;
        num_proj      = 0;
        pix_size      = 1;
        KV            = 300;
        CS            = 2.7;
        AC            = 0.07;
        R             = NULL;
        t             = NULL;
        w             = NULL;
        def           = NULL;
    }

    ~Tomogram() {
        free_array( R   );
        free_array( t   );
        free_array( w   );
        free_array( def );
    }

    void read(IO::TxtParser&parser) {

        parser.get_value( tomo_id ,"tomo_id"   );
        parser.get_value( tomo_dim,"tomo_size" );
        parser.get_str  ( stk_name,"stack_file");
        parser.get_value( stk_dim ,"stack_size");
        parser.get_value( pix_size,"pix_size"  );
        parser.get_value( KV      ,"kv"        );
        parser.get_value( CS      ,"cs"        );
        parser.get_value( AC      ,"ac"        );
        parser.get_value( num_proj,"num_proj"  );

        tomo_center(0) = ((float)tomo_dim.x)/2;
        tomo_center(1) = ((float)tomo_dim.y)/2;
        tomo_center(2) = ((float)tomo_dim.z)/2;

        stk_center(0) = ((float)stk_dim.x)/2;
        stk_center(1) = ((float)stk_dim.y)/2;
        stk_center(2) = ((float)stk_dim.z)/2;

        allocate(num_proj);

        for(int i=0;i<num_proj;i++) {

            V3f eu;
            char*buf = parser.read_line_raw();
            t[i](2) = 0;

            int n = sscanf(buf,"%f %f %f %f %f %f %f %f %f %f %f %f %f",
                           &eu(0),&eu(1),&eu(2),&t[i](0),&t[i](1),&w[i],
                           &def[i].U,&def[i].V,&def[i].angle,
                           &def[i].Bfactor,&def[i].ExpFilt,
                           &def[i].max_res,&def[i].score);

            if( n != 13 ) {
                fprintf(stderr,"Truncated tomogram file.\n");
                exit(0);
            }

            eu *= M_PI/180;
            Math::eZYZ_Rmat(R[i],eu);

        }
    }

    void print() {
        printf("  Tomo ID:    %d\n",tomo_id);
        printf("  Tomo size:  [%d %d %d]\n",tomo_dim.x,tomo_dim.y,tomo_dim.z);
        printf("  Stack file: %s\n",stk_name);
        printf("  Stack size: [%d %d %d]\n",stk_dim.x,stk_dim.y,stk_dim.z);
        printf("  Num Proj:   %d\n",num_proj);
        printf("  Pix size:   %.3f\n",pix_size);
        printf("  KV:         %.1f\n",KV);
        printf("  CS:         %.2f\n",CS);
        printf("  AC:         %.3f\n",AC);
        printf("  Projections:\n");
        for(int i=0;i<num_proj;i++) {
            printf("    %2d:  %7.4f %7.4f %7.4f   %7.2f\n",      i,R[i](0,0),R[i](0,1),R[i](0,2),t[i](0));
            printf( "         %7.4f %7.4f %7.4f   %7.2f\n",        R[i](1,0),R[i](1,1),R[i](1,2),t[i](1));
            printf( "         %7.4f %7.4f %7.4f             %f\n", R[i](2,0),R[i](2,1),R[i](2,2),w[i]);
        }
        printf("  Defocus:\n");
        for(int i=0;i<num_proj;i++) {
            printf("    %2d:  %10.2f %10.2f %8.3f %7.2f %7.2f %8.4f\n",i,def[i].U,def[i].V,def[i].angle,def[i].Bfactor,def[i].ExpFilt,def[i].max_res);
        }
    }
    
    bool check() {
		bool rslt = check_exists();
		
		if( rslt ) {
			rslt &= check_size();
			rslt &= check_mode();
			rslt &= check_apix();
		}
		
		return rslt;
	}

protected:
    void allocate(int K) {
        R   = new M33f   [K];
        t   = new V3f    [K];
        w   = new float  [K];
        def = new Defocus[K];
    }
    
    bool check_exists() {
		if( ~IO::exists(stk_name) ) {
			fprintf(stderr,"File %s not found or cannot be read.\n",stk_name);
			return false;
		}
		return true;
	}
	
	bool check_size() {
		uint32 x,y,z;
		Mrc::read_size(x,y,z,stk_name);
		if( x != stk_dim.x || y != stk_dim.y || z != stk_dim.z ) {
			fprintf(stderr,"file %s: Different size (%d,%d,%d) != (%d,%d,%d).\n",stk_name,x,y,z,stk_dim.x,stk_dim.y,stk_dim.z);
			return false;
		}
		return true;
	}
	
	bool check_mode() {
		if( !Mrc::is_mode_float(stk_name) ) {
			fprintf(stderr,"file %s: Unsupported MRC mode.\n",stk_name);
			return false;
		}
		return true;
	}
	
	bool check_apix() {
		float stk_pix_size = Mrc::get_apix(stk_name);
		if( abs(stk_pix_size-pix_size) < SUSAN_FLOAT_TOL ) {
			fprintf(stderr,"file %s: Different pixel size %f != %f.\n",stk_name,stk_pix_size,pix_size);
			return false;
		}
		return true;
	}
};

class Tomograms {
public:
    uint32   num_tomo;
    uint32   num_proj;

protected:
    Tomogram *tomos;

public:
    Tomograms(const char*filename) {
        tomos = NULL;

        IO::TxtParser parser(filename,"tomostxt");
        parser.get_value(num_tomo,"num_tomos");
        parser.get_value(num_proj,"num_projs");

        tomos = new Tomogram[num_tomo];

        for(int i=0;i<num_tomo;i++)
            tomos[i].read(parser);
    }

    ~Tomograms() {
        free_array(tomos);
    }

    Tomogram& operator[](int ix) {
        if( ix >= num_tomo ) {
            fprintf(stderr,"Trying to access tomogram %d from %d.\n",ix,num_tomo);
            exit(0);
        }
        return tomos[ix];
    }

    Tomogram& at(int ix) {
        if( ix >= num_tomo ) {
            fprintf(stderr,"Trying to access tomogram %d from %d.\n",ix,num_tomo);
            exit(0);
        }
        return tomos[ix];
    }

    void print() {
        printf("Num Tomos: %d\n",num_tomo);
        printf("Max Projs: %d\n",num_proj);
        for(int i=0;i<num_tomo;i++) {
            printf("Tomogram %d\n",i);
            tomos[i].print();
        }
    }
    
    bool check() {
		bool rslt = true;
		for(int i=0;i<num_tomo;i++) {
            rslt &= tomos[i].check();
        }
        return rslt;
	}

};

#endif /// TOMOGRAM_H


