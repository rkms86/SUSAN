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
        mexErrMsgTxt("[fsc_get_core] One output required");
    }

    if( nIn  != 3 ) {
        mexErrMsgTxt("[fsc_get_core] Three inputs required");
    }
    
    float *num,*denA,*denB;
    double*output,*countA,*countB;
    mwSize X,Y,Z;
  
    if( is_single(pIn[0]) && is_single(pIn[1]) && is_single(pIn[2]) ) {
        
        get_array(pIn[0],num,X,Y,Z);
        get_array(pIn[1],denA);
        get_array(pIn[2],denB);
        mwSize L = X/2 + 1;
        allocate(pOut[0], L, 1, 1, mxDOUBLE_CLASS, mxREAL);
        get_array(pOut[0],output);
        countA= (double*)malloc(L*sizeof(double));
        countB= (double*)malloc(L*sizeof(double));
        memset((void*)output,0,L*sizeof(double));
        memset((void*)countA,0,L*sizeof(double));
        memset((void*)countB,0,L*sizeof(double));
        
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
                    int r1 = r0 + 1;

                    float w1 = R - floor(R);
                    float w0 = 1 - w1;

                    float val_n = (double)num [j+i*X+k*X*Y];
                    float val_a = (double)denA[j+i*X+k*X*Y];
                    float val_b = (double)denB[j+i*X+k*X*Y];

                    if( r0 >= 0 && r0 < L ) {
                        output[r0] += val_n;
                        countA[r0] += val_a;
                        countB[r0] += val_b;
                    }

                    //if( r1 >= 0 && r1 < L ) {
                    //    output[r1] += val_n*w1;
                    //    countA[r1] += val_a*w1;
                    //    countB[r1] += val_b*w1;
                    //}
                }
            }
        }
        
        for(i=0;i<L;i++) {
            output[i] = output[i]/sqrt(countA[i]*countB[i]);
        }
                
        free(countA);
        free(countB);
    }
    else {
        mexErrMsgTxt("[fsc_get_core] Wrong inputs.");
    }  
}
