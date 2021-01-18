#define MW_NEEDS_VERSION_H /// For Matlab > R2018b
#include "mex.h"
#include "math.h"

#include "matlab.h"
#include "datatypes.h"

#include "math_cpu.h"

using namespace Matlab;

#define MEX_FUNC_NAME "Aligner.list_cone_search"

bool check_input_types(const mxArray *pIn[]) {
    bool rslt = true;
    rslt = rslt && is_single(pIn[0]); // cone_range
    rslt = rslt && is_single(pIn[1]); // cone_step
    rslt = rslt && is_uint32(pIn[2]); // refine_level
    rslt = rslt && is_uint32(pIn[3]); // refine_factor
    rslt = rslt && is_char  (pIn[4]); // pseudo_symmetry
    return rslt;
}

/// [pts,lvl] = Aligner_list_cone_search(cone_range,cone_step,refine_level,refine_factor,pseudo_symmetry);
void mexFunction(int nOut, mxArray *pOut[], int nIn, const mxArray *pIn[]) {

    if( nOut != 2 ) {
        mexErrMsgTxt("[" MEX_FUNC_NAME "] Two outputs required");
    }

    if( nIn  != 5 ) {
        mexErrMsgTxt("[" MEX_FUNC_NAME "] Five inputs required");
    }
    
    if( check_input_types( pIn ) ) {
        char   *pseudo_sym;
        mwSize L;
        
        Math::AngleProvider angles;
        angles.cone_range    = get_scalar_single(pIn[0]);
        angles.cone_step     = fmax(get_scalar_single(pIn[1]),1);
        angles.inplane_range = 0;
        angles.inplane_step  = 1;
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
        
        single *po_x,*po_y,*po_z;
        V3f    pt_z,pt;
        M33f   R;
        uint32 *plvl;
        
        allocate_real(pOut[0], angle_count, 3, 1, mxSINGLE_CLASS);
        allocate_real(pOut[1], 1, angle_count, 1, mxUINT32_CLASS);
        get_array(pOut[0], po_x);
        get_array(pOut[1], plvl);
        po_y = po_x+angle_count;
        po_z = po_y+angle_count;
        
        pt_z(0) = 0;
        pt_z(1) = 0;
        pt_z(2) = 1;
        angle_count = 0;
        uint32 lvl_count = 1;
        for( angles.levels_init(); angles.levels_available(); angles.levels_next() ) {
            for( angles.sym_init(); angles.sym_available(); angles.sym_next() ) {
                for( angles.cone_init(); angles.cone_available(); angles.cone_next() ) {
                    for( angles.inplane_init(); angles.inplane_available(); angles.inplane_next() ) {
                        angles.get_current_R(R);
                        pt = R*pt_z;
                        po_x[angle_count] = pt(0);
                        po_y[angle_count] = pt(1);
                        po_z[angle_count] = pt(2);
                        plvl[angle_count] = lvl_count;
                        angle_count++;
                    }
                }
            }
            lvl_count++;
            pt_z(2) += 0.1;
        }
    }
    else {
        mexErrMsgTxt("[" MEX_FUNC_NAME "] Wrong input types.");
    }  
}
