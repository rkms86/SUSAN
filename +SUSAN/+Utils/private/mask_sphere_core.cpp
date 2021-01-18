#define MW_NEEDS_VERSION_H /// For Matlab > R2018b
#include "mex.h"
#include "math.h"

#include "matlab.h"
#include "mrc.h"
#include "datatypes.h"

#include <cstring>

using namespace Matlab;

void mexFunction(int nOut, mxArray *pOut[], int nIn, const mxArray *pIn[]) {

    if( nOut != 1 ) {
        mexErrMsgTxt("[mask_sphere_core] One output required");
    }

    if( nIn  != 3 ) {
        mexErrMsgTxt("[mask_sphere_core] Three inputs required");
    }
    
    if( is_single(pIn[0]) && is_single(pIn[1]) && is_single(pIn[2]) ) {
    
        float *out;
        float rad,siz,*center;
        mwSize X,Y,Z;
        
        rad = get_scalar_single(pIn[0]);
        siz = get_scalar_single(pIn[1]);
        get_array(pIn[2],center,X,Y,Z);
        
        mwSize L = (mwSize)round(siz);
        allocate(pOut[0], L, L, L, mxSINGLE_CLASS, mxREAL);
        get_array(pOut[0],out);
        
        float x,y,z;
        
        mwSize i,j,k;
        
        for(k=0;k<L;k++) {

            z = ((float)k) - center[2];

            for(i=0;i<L;i++) {

                x = ((float)i) - center[0];

                for(j=0;j<L;j++) {

                    y = ((float)j) - center[1];

                    float R = sqrt( x*x + y*y + z*z );
                    R = fmin( fmax( rad - R , 0.0), 1.0 );
                    out[ j + i*L + k*L*L ] = R;
                    
                }
            }
        }
    }
    else {
        mexErrMsgTxt("[mask_sphere_core] Wrong inputs.");
    }  
}
