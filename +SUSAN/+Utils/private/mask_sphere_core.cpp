/*
 * This file is part of the Substack Analysis (SUSAN) framework.
 * Copyright (c) 2018-2021 Ricardo Miguel Sanchez Loayza.
 * Max Planck Institute of Biophysics
 * Department of Structural Biology - Kudryashev Group.
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
