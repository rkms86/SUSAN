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

#define FUNC_NAME "Particles.Geom.rot_shift"


/// [new_euZYZ, new_T] = Particles.Geom.rot_shift(R,t,old_euZYZ, old_T);
void mexFunction(int nOut, mxArray *pOut[], int nIn, const mxArray *pIn[]) {

    if( nOut != 2 ) {
        mexErrMsgTxt("[" FUNC_NAME "] Two outputs required");
    }

    ///
    if( nIn  != 4 ) {
        mexErrMsgTxt("[" FUNC_NAME "] Four inputs required");
    }

    if( is_single(pIn[0]) && is_single(pIn[1]) && is_single(pIn[2]) && is_single(pIn[3]) ) {

        mwSize   X,Y,Z;
        float    *po_eu,*pi_eu;
        float    *po_tt,*pi_tt;
        float    *tt,*Rcol;
        mwSize   num_ptlcs;


        /// Read inputs:
        get_array(pIn[0],Rcol);
        get_array(pIn[1],tt);
        get_array(pIn[2],pi_eu);
        get_array(pIn[3],pi_tt,X,Y,Z);
        num_ptlcs = X;

        /// Allocate output:
        allocate_and_get(po_eu,pOut[0],num_ptlcs,3,1);
        allocate_and_get(po_tt,pOut[1],num_ptlcs,3,1);

        /// Convert:
        V3f  v, eu, vout;
        M33f Rin,Rout,R;
        v(0) = tt[0];
        v(1) = tt[1];
        v(2) = tt[2];
        R(0,0) = Rcol[0];
        R(1,0) = Rcol[1];
        R(2,0) = Rcol[2];
        R(0,1) = Rcol[3];
        R(1,1) = Rcol[4];
        R(2,1) = Rcol[5];
        R(0,2) = Rcol[6];
        R(1,2) = Rcol[7];
        R(2,2) = Rcol[8];

        for(int x=0;x<num_ptlcs;x++) {
            eu(0) = pi_eu[x    ];
            eu(1) = pi_eu[x+  X];
            eu(2) = pi_eu[x+2*X];

            eu *= M_PI/180;
            Math::eZYZ_Rmat(Rin,eu);
            Rout = R*Rin;
            Math::Rmat_eZYZ(eu,Rout);
            eu *= 180/M_PI;

            vout = Rout.transpose()*v;

            po_eu[x    ] = eu(0);
            po_eu[x+  X] = eu(1);
            po_eu[x+2*X] = eu(2);

            po_tt[x    ] = vout(0)+pi_tt[x    ];
            po_tt[x+  X] = vout(1)+pi_tt[x+  X];
            po_tt[x+2*X] = vout(2)+pi_tt[x+2*X];

        }

    }
    else {
        mexErrMsgTxt("[" FUNC_NAME "] Wrong Inputs' types.");
    }
}



