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
#include "datatypes.h"
#include "math_cpu.h"

#include "Eigen/Geometry"


using namespace Eigen;
using namespace Matlab;

#define FUNC_NAME "ParticlesInfo.euDYN_2_euZYZ"


/// euZYZ = euDYN_2_euZYZ(eu_dynamo)
void mexFunction(int nOut, mxArray *pOut[], int nIn, const mxArray *pIn[]) {

    if( nOut != 1 ) {
        mexErrMsgTxt("[" FUNC_NAME "] One output required");
    }

    ///
    if( nIn  != 1 ) {
        mexErrMsgTxt("[" FUNC_NAME "] One input required");
    }

    if( is_single(pIn[0]) ) {

        mwSize   X,Y,Z;
        float    *p_euDYN;
        float    *p_euZYZ;

        /// Read inputs:
        get_array(pIn[0],p_euDYN,X,Y,Z);
        if( Y != 3 || Z != 1 ) {
			mexErrMsgTxt("[" FUNC_NAME "] Wrong angles matrix dimensions");
		}

		/// Allocate output:
        allocate_and_get(p_euZYZ,pOut[0],X,3,1);

        /// Convert:
        V3f  euZXZ, euZYZ;
        M33f R;

        for(int z=0;z<Z;z++) {

            for(int x=0;x<X;x++) {
                euZXZ(2) = p_euDYN[x    ];
                euZXZ(1) = p_euDYN[x+  X];
                euZXZ(0) = p_euDYN[x+2*X];

                euZXZ *= M_PI/180;
                Math::eZXZ_Rmat(R,euZXZ);
                Math::Rmat_eZYZ(euZYZ,R);
                euZYZ *= 180/M_PI;

                p_euZYZ[x    ] = euZYZ(0);
                p_euZYZ[x+  X] = euZYZ(1);
                p_euZYZ[x+2*X] = euZYZ(2);
            }
        }

    }
    else {
        mexErrMsgTxt("[" FUNC_NAME "] Wrong Inputs' types.");
    }
}



