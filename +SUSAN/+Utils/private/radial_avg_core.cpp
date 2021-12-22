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
        mexErrMsgTxt("[radial_avg_core] One output required");
    }

    if( nIn  != 1 ) {
        mexErrMsgTxt("[radial_avg_core] One inputs required");
    }
    
    float *num;
    double*output,*count;
    mwSize X,Y,Z;
  
    if( is_single(pIn[0]) ) {
        
        get_array(pIn[0],num,X,Y,Z);
        mwSize L = X/2 + 1;
        allocate(pOut[0], L, 1, 1, mxDOUBLE_CLASS, mxREAL);
        get_array(pOut[0],output);
        count= (double*)malloc(L*sizeof(double));
        memset((void*)output,0,L*sizeof(double));
        memset((void*)count,0,L*sizeof(double));
        
        float c_x = float(X)/2;
        float c_y = float(Y)/2;
        float c_z = float(Z)/2;

        float x,y,z;
        
        mwSize i,j,k;
        
        for(k=0;k<Z;k++) {

            z = ((float)k) - c_z;

            for(i=0;i<X;i++) {

                x = ((float)i) - c_x;

                for(j=0;j<Y;j++) {

                    y = ((float)j) - c_y;

                    float R = sqrt( x*x + y*y + z*z );

                    int r0 = (int)(floor(R));

                    float val_n = (double)num [j+i*X+k*X*Y];

                    if( r0 >= 0 && r0 < L ) {
                        output[r0] += val_n;
                        count [r0] += 1.0;
                    }
                }
            }
        }
        
        for(i=0;i<L;i++) {
            output[i] = output[i]/count[i];
        }
        
        free(count);
    }
    else {
        mexErrMsgTxt("[radial_avg_core] Wrong inputs.");
    }  
}
