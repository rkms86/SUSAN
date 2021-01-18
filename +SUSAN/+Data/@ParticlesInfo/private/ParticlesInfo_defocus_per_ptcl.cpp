#define MW_NEEDS_VERSION_H /// For Matlab > R2018b
#include "mex.h"
#include "math.h"

#include "matlab.h"
#include "datatypes.h"
#include "math_cpu.h"

#include "Eigen/Geometry"

using namespace Eigen;
using namespace Matlab;

#define FUNC_NAME "ParticlesInfo.defocus_per_ptcl"

/// ptcl_defocus = defocus_per_ptcl(tomo_cix,position,proj_eZYZ,tomo_defocus);
void mexFunction(int nOut, mxArray *pOut[], int nIn, const mxArray *pIn[]) {

    if( nOut != 1 ) {
        mexErrMsgTxt("[" FUNC_NAME "] One output required");
    }

    ///
    if( nIn  != 4 ) {
        mexErrMsgTxt("[" FUNC_NAME "] Four inputs required");
    }

    if( is_uint32(pIn[0]) && is_single(pIn[1]) && is_single(pIn[2]) && is_single(pIn[3]) ) {

        mwSize X,Y,Z;
        int    num_tomos, num_projs, num_ptcl;
        uint32 *p_tomo_ix;
        float  *p_pos,*p_euZYZ,*p_shift,*p_def_in,*p_def_out;

        /// Read inputs:
        get_array(pIn[0],p_tomo_ix,X,Y,Z);
        num_ptcl = X;
        if( Y != 1 || Z != 1 ) {
            char tmp[1024];
            sprintf(tmp,"[" FUNC_NAME "] Arg1 (tomo_cix) wrong size [%d,%d,%d], should be [%d 1 1]",X,Y,Z,num_ptcl);
            mexErrMsgTxt(tmp);
        }

        get_array(pIn[1],p_pos,X,Y,Z);
        if( X != num_ptcl || Y != 3 || Z != 1 ) {
            char tmp[1024];
            sprintf(tmp,"[" FUNC_NAME "] Arg2 (position) wrong size [%d,%d,%d], should be [%d 3 1]",X,Y,Z,num_ptcl);
            mexErrMsgTxt(tmp);
        }

        get_array(pIn[2],p_euZYZ,X,Y,Z);
        num_projs = X;
        num_tomos = Z;
        if( Y != 3 ) {
            char tmp[1024];
            sprintf(tmp,"[" FUNC_NAME "] Arg3 (tomo eZYZ) wrong size [%d,%d,%d], should be [%d 3 %d]",X,Y,Z,X,Z);
            mexErrMsgTxt(tmp);
        }

        get_array(pIn[3],p_def_in,X,Y,Z);
        if( X != num_projs || Y != 7 || Z != num_tomos ) {
            char tmp[1024];
            sprintf(tmp,"[" FUNC_NAME "] Arg4 (defocus) wrong size [%d,%d,%d], should be [%d 7 %d]",X,Y,Z,num_projs,num_tomos);
            mexErrMsgTxt(tmp);
        }
        
        /// Allocate output:
        allocate_and_get(p_def_out,pOut[0],num_projs,7,num_ptcl);

        /// Expand Defocus:
        V3f    pos_tomo,pos_proj;
        V3f    euZYZ;
        V3f    shift;
        M33f   R;
        uint32 tomo_ix;

		float  *cur_tomo_def;
        float  *cur_tomo_eZYZ;
        float  *cur_ptcl_def;
        float  d_def;
        
        for(int ptcl=0;ptcl<num_ptcl;ptcl++) {

            tomo_ix = p_tomo_ix[ptcl];

            pos_tomo(0)  = p_pos[ptcl           ];
            pos_tomo(1)  = p_pos[ptcl+  num_ptcl];
            pos_tomo(2)  = p_pos[ptcl+2*num_ptcl];

            cur_tomo_def  = p_def_in  + tomo_ix*(num_projs*7);
            cur_tomo_eZYZ = p_euZYZ   + tomo_ix*(num_projs*3);
            cur_ptcl_def  = p_def_out +    ptcl*(num_projs*7);

            for(int proj=0;proj<num_projs;proj++) {
                euZYZ(0) = cur_tomo_eZYZ[proj    ];
                euZYZ(1) = cur_tomo_eZYZ[proj+  num_projs];
                euZYZ(2) = cur_tomo_eZYZ[proj+2*num_projs];
                euZYZ   *= M_PI/180;
                Math::eZYZ_Rmat(R,euZYZ);
                pos_proj = R*pos_tomo;
                cur_ptcl_def[proj            ] = cur_tomo_def[proj            ] - pos_proj(2); // U
                cur_ptcl_def[proj+  num_projs] = cur_tomo_def[proj+  num_projs] - pos_proj(2); // V
                cur_ptcl_def[proj+2*num_projs] = cur_tomo_def[proj+2*num_projs]; // angle
                cur_ptcl_def[proj+3*num_projs] = cur_tomo_def[proj+3*num_projs]; // BFactor
                cur_ptcl_def[proj+4*num_projs] = cur_tomo_def[proj+4*num_projs]; // Exposure Filter
                cur_ptcl_def[proj+5*num_projs] = cur_tomo_def[proj+5*num_projs]; // Maximum Resolution
                cur_ptcl_def[proj+6*num_projs] = cur_tomo_def[proj+6*num_projs]; // Score
            }
        }

    }
    else {
        mexErrMsgTxt("[" FUNC_NAME "] Wrong Inputs' types.");
    }
}
