#define MW_NEEDS_VERSION_H /// For Matlab > R2018b
#include "mex.h"
#include "math.h"

#include "matlab.h"
#include "datatypes.h"

#include "math_cpu.h"

using namespace Matlab;

#define MEX_FUNC_NAME "Aligner.list_inplane_search"

bool check_input_types(const mxArray *pIn[]) {
    bool rslt = true;
    rslt = rslt && is_single(pIn[0]); // inplane_range
    rslt = rslt && is_single(pIn[1]); // inplane_step
    rslt = rslt && is_uint32(pIn[2]); // refine_level
    rslt = rslt && is_uint32(pIn[3]); // refine_factor
    rslt = rslt && is_char  (pIn[4]); // pseudo_symmetry
    return rslt;
}

/// [pts_x, pts_y,lvl] = Aligner_list_inplane_search(cone_range,cone_step,refine_level,refine_factor,pseudo_symmetry);
void mexFunction(int nOut, mxArray *pOut[], int nIn, const mxArray *pIn[]) {

    if( nOut != 3 ) {
        mexErrMsgTxt("[" MEX_FUNC_NAME "] Three outputs required");
    }

    if( nIn  != 5 ) {
        mexErrMsgTxt("[" MEX_FUNC_NAME "] Five inputs required");
    }
    
    if( check_input_types( pIn ) ) {
        char   *pseudo_sym;
        mwSize L;
        
        Math::AngleProvider angles;
        angles.cone_range    = 0;
        angles.cone_step     = 1;
        angles.inplane_range = get_scalar_single(pIn[0]);
        angles.inplane_step  = fmax(get_scalar_single(pIn[1]),1);
        angles.refine_level  = get_scalar_uint32(pIn[2]);
        angles.refine_factor = get_scalar_uint32(pIn[3]);
        get_array(pIn[4],pseudo_sym,L);
        
        /// Get Total Angles:
        uint32 angle_count = 0;
        for( angles.levels_init(); angles.levels_available(); angles.levels_next() ) {
            for( angles.sym_init(); angles.sym_available(); angles.sym_next() ) {
                for( angles.cone_init(); angles.cone_available(); angles.cone_next() ) {
                    for( angles.inplane_init(); angles.inplane_available(); angles.inplane_next() ) {
                        angle_count++;
                    }
                }
            }
        }
        
        single *ptmp,*po_x,*po_y;
        V3f    pt_x,pt;
        M33f   R;
        uint32 *plvl;
        
        allocate_real(pOut[0], angle_count, 2, 1, mxSINGLE_CLASS);
        allocate_real(pOut[1], angle_count, 2, 1, mxSINGLE_CLASS);
        allocate_real(pOut[2], 1, angle_count, 1, mxUINT32_CLASS);
        get_array(pOut[0], ptmp);
        memset(ptmp,0,angle_count*sizeof(single));
        po_x = ptmp+angle_count;
        get_array(pOut[1], ptmp);
        memset(ptmp,0,angle_count*sizeof(single));
        po_y = ptmp+angle_count;
        get_array(pOut[2], plvl);
        
        pt_x(0) = 1+0.1*angles.refine_level;
        pt_x(1) = 0;
        pt_x(2) = 0;
        angle_count = 0;
        uint32 lvl_count = 1;
        for( angles.levels_init(); angles.levels_available(); angles.levels_next() ) {
            for( angles.sym_init(); angles.sym_available(); angles.sym_next() ) {
                for( angles.cone_init(); angles.cone_available(); angles.cone_next() ) {
                    for( angles.inplane_init(); angles.inplane_available(); angles.inplane_next() ) {
                        angles.get_current_R(R);
                        pt = R*pt_x;
                        po_x[angle_count] = pt(0);
                        po_y[angle_count] = pt(1);
                        plvl[angle_count] = lvl_count;
                        angle_count++;
                    }
                }
            }
            lvl_count++;
            pt_x(0) = pt_x(0)-0.1;
        }
    }
    else {
        mexErrMsgTxt("[" MEX_FUNC_NAME "] Wrong input types.");
    }  
}
