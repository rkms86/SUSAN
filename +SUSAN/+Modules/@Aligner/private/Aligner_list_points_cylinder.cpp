#define MW_NEEDS_VERSION_H /// For Matlab > R2018b
#include "mex.h"
#include "math.h"

#include "matlab.h"
#include "datatypes.h"

#include "math_cpu.h"

using namespace Matlab;

bool check_input_types(const mxArray *pIn[]) {
    bool rslt = true;
    rslt = rslt && is_single(pIn[0]); // offset_range
    rslt = rslt && is_single(pIn[1]); // offset_step
    return rslt;
}

/// [P] = Aligner_list_points_cylinder(offset_range,offset_step);
void mexFunction(int nOut, mxArray *pOut[], int nIn, const mxArray *pIn[]) {

    if( nOut != 1 ) {
        mexErrMsgTxt("[Aligner.list_points_cylinder] One output required");
    }

    if( nIn  != 2 ) {
        mexErrMsgTxt("[Aligner.list_points_cylinder] Two inputs required");
    }
    
    if( check_input_types( pIn ) ) {
        single *offset_range_arr;
        single *offset_step_arr;
        
        get_array(pIn[0],offset_range_arr);
        get_array(pIn[1],offset_step_arr);
        
        uint32 L;
        
        Vec3 *points = Math::create_points_cylinder(L,offset_range_arr[0],offset_range_arr[1],offset_range_arr[2],offset_step_arr[0]);
        
        single *P;
        allocate_real(pOut[0], 3, L, 1, mxSINGLE_CLASS);
        get_array(pOut[0], P);
        
        for(uint32 i=0;i<L;i++) {
            P[3*i  ] = points[i].x;
            P[3*i+1] = points[i].y;
            P[3*i+2] = points[i].z;
        }
        
        delete [] points;
    }
    else {
        mexErrMsgTxt("[Aligner.list_points_cylinder] Wrong input types.");
    }  
}
