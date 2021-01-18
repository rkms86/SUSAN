#define MW_NEEDS_VERSION_H /// For Matlab > R2018b
#include "mex.h"
#include "math.h"

#include "matlab.h"

using namespace Matlab;

#define FUNC_NAME "Particles.Geom.select_min_dist"

typedef struct {
    float x;
    float y;
    float z;
    float w;
    float i;
} Point_st;

/// cur_idx = ParticlesInfo_discard_min_distance(data,min_dist)
void mexFunction(int nOut, mxArray *pOut[], int nIn, const mxArray *pIn[]) {

    ///
    if( nOut != 1 ) {
        mexErrMsgTxt("[" FUNC_NAME "] No output required");
    }

    ///
    if( nIn  != 2 ) {
        mexErrMsgTxt("[" FUNC_NAME "] Two inputs required");
    }

    /// Validate Inputs' types:
    if( !is_single(pIn[0]) || !is_single(pIn[1]) )
        mexErrMsgTxt("[" FUNC_NAME "] Input must be a single matrix string.");

    /// Get Inputs:
    mwSize X,N,Z;
    float *p_tmp;
    get_array(pIn[0],p_tmp,X,N,Z);
    if( X != 5 || Z != 1 ) {
        char tmp[1024];
        sprintf(tmp,"[" FUNC_NAME "] Arg1 wrong size [%d,%d,%d], should be [5 %d 1]",X,N,Z,N);
        mexErrMsgTxt(tmp);
    }
    Point_st *pt = (Point_st*)p_tmp;
    
    float min_dist_2 = get_scalar_single(pIn[1]);
    min_dist_2 = min_dist_2*min_dist_2;
    
    /// Allocate Output
    uint32_t *p_out;
    allocate_and_get(p_out,pOut[0],N,1,1);

    /// Calculate distances:
    for(int cur_idx=0;cur_idx<N;cur_idx++) {
        
        int out_i = (int)pt[cur_idx].i;
        p_out[out_i] = 1;
        
        for(int prv_idx=0;prv_idx<cur_idx;prv_idx++) {
            
            int in_i = (int)pt[prv_idx].i;
            if( p_out[in_i] > 0 ) {
                float x = pt[cur_idx].x - pt[prv_idx].x;
                float y = pt[cur_idx].y - pt[prv_idx].y;
                float z = pt[cur_idx].z - pt[prv_idx].z;
                float cur_dist_2 = x*x + y*y + z*z;
                if( cur_dist_2 < min_dist_2 ) {
                    p_out[out_i] = 0;
                    break;
                }
            }
            
        }
        
    }
    
}
