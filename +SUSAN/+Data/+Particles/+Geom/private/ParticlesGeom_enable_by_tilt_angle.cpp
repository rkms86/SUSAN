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
#include "datatypes.h"
#include "math_cpu.h"

#include "Eigen/Geometry"


using namespace Eigen;
using namespace Matlab;

#include <iostream>

#define FUNC_NAME "ParticlesGeom.select_by_tilt_angle"

void get_w(float*p_w,const float*p_euZYZ,const float max_angle,const int tix, const uint32 num_proj) {
    const float*local_eu = p_euZYZ + tix*num_proj*3;
    V3f euZYZ,vZ,vP;
    M33f R;
    vZ(0) = 0;
    vZ(1) = 0;
    vZ(2) = 1;

    for(int i=0;i<num_proj;i++) {
        euZYZ(0) = local_eu[i           ];
        euZYZ(1) = local_eu[i+  num_proj];
        euZYZ(2) = local_eu[i+2*num_proj];
        euZYZ *= M_PI/180;
        Math::eZYZ_Rmat(R,euZYZ);
        vP = R*vZ;
        float cur_ang = acos( vP.dot(vZ) );
        if( cur_ang < max_angle ) {
            p_w[i] = 1;
        }
        else  {
            p_w[i] = 0;
        }
    }
}

/// w = select_by_tilt_angle(tomo_cix,euZYZ,max_angle)
void mexFunction(int nOut, mxArray *pOut[], int nIn, const mxArray *pIn[]) {

    if( nOut != 1 ) {
        mexErrMsgTxt("[" FUNC_NAME "] One output required");
    }

    ///
    if( nIn  != 3 ) {
        mexErrMsgTxt("[" FUNC_NAME "] Two inputs required");
    }

    if( is_uint32(pIn[0]) && is_single(pIn[1]) && is_single(pIn[2]) ) {

        mwSize   X,Y,Z,num_ptcl,num_proj,num_tomo;
        uint32   *p_tomo_cix;
        float    *p_euZYZ;
        float    *p_w;
        
        /// Read inputs:
        get_array(pIn[0],p_tomo_cix,num_ptcl,Y,Z);
        if( Y != 1 || Z != 1 ) {
			mexErrMsgTxt("[" FUNC_NAME "] tomo cix must be a vector");
		}

        get_array(pIn[1],p_euZYZ,num_proj,Y,num_tomo);
        if( Y != 3 ) {
			mexErrMsgTxt("[" FUNC_NAME "] Angles must be euler ZYZ triplets");
		}
		
		float angle_range = get_scalar_single(pIn[2]);
		angle_range = angle_range*M_PI/180;

        /// Allocate output:
        allocate_and_get(p_w,pOut[0],num_proj,1,num_ptcl);

        /// Convert:
        float cur_w[num_proj];
        int cur_tix=0;
        get_w(cur_w,p_euZYZ,angle_range,cur_tix,num_proj);

        for(int ix=0;ix<num_ptcl;ix++) {

            if( p_tomo_cix[ix] != cur_tix ) {
                cur_tix = p_tomo_cix[ix];
                get_w(cur_w,p_euZYZ,angle_range,cur_tix,num_proj);
            }
            
            memcpy(p_w+ix*num_proj,cur_w,sizeof(float)*num_proj);
        }

    }
    else {
        mexErrMsgTxt("[" FUNC_NAME "] Wrong Inputs' types.");
    }
}



