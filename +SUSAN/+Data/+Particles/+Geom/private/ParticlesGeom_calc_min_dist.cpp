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

using namespace Matlab;

#define FUNC_NAME "Particles.Geom.calc_min_dist"

typedef struct {
    float x;
    float y;
    float z;
} Point_st;

/// cur_idx = ParticlesInfo_discard_min_distance(data,min_dist)
void mexFunction(int nOut, mxArray *pOut[], int nIn, const mxArray *pIn[]) {

    ///
    if( nOut != 1 ) {
        mexErrMsgTxt("[" FUNC_NAME "] No output required");
    }

    ///
    if( nIn  != 1 ) {
        mexErrMsgTxt("[" FUNC_NAME "] One input required");
    }

    /// Validate Inputs' types:
    if( !is_single(pIn[0]) )
        mexErrMsgTxt("[" FUNC_NAME "] Input must be a single matrix string.");

    /// Get Inputs:
    mwSize X,N,Z;
    float *p_tmp;
    get_array(pIn[0],p_tmp,X,N,Z);
    if( X != 3 || Z != 1 ) {
        char tmp[1024];
        sprintf(tmp,"[" FUNC_NAME "] Arg1 wrong size [%d,%d,%d], should be [3 %d 1]",X,N,Z,N);
        mexErrMsgTxt(tmp);
    }
    Point_st *pt = (Point_st*)p_tmp;
    
    /// Allocate Output
    float *p_dist;
    allocate_and_get(p_dist,pOut[0],N,1,1);

    /// Calculate distances:
    for(int idx_a=0;idx_a<N;idx_a++) {
        
        p_dist[idx_a] = 999999.9;
        
        for(int idx_b=0;idx_b<N;idx_b++) {
            
            if( idx_a != idx_b ) {
                float x = pt[idx_b].x - pt[idx_a].x;
                float y = pt[idx_b].y - pt[idx_a].y;
                float z = pt[idx_b].z - pt[idx_a].z;
                float cur_dist = sqrt( x*x + y*y + z*z );
                
                if( cur_dist < p_dist[idx_a] )
                    p_dist[idx_a] = cur_dist;
                
            }
        }
        
    }
    
}
