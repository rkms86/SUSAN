#ifndef ANGLES_SYMMETRY_H
#define ANGLES_SYMMETRY_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

#include "datatypes.h"
#include "math_cpu.h"

using namespace Eigen;
using namespace Math;

class AnglesSymmetry {

public:
    static M33f*get_rotation_list(uint32 &N, const char*sym_str) {
        N = 0;
        if( strcmp(sym_str,"cone_flip") == 0 || strcmp(sym_str,"y_180") == 0 ) {
            N = 2;
            return cone_flip();
        }
        else if( strcmp(sym_str,"cbo") == 0 ) {
            N = 24;
            return cuboctahedral();
        }
        else if( strcmp(sym_str,"c1") == 0 || strcmp(sym_str,"none") == 0 ) {
            N = 1;
            return none();
        }
        else if( sym_str[0] == 'c' ) {
            N = atoi(sym_str+1);
            return Cn(N);
        }
        else {
            return NULL;
        }
    }

protected:
    static M33f*none() {
        M33f*rslt = new M33f[1];
        V3f eu;
        eu(0) = 0;
        eu(1) = 0;
        eu(2) = 0;
        Math::eZYZ_Rmat(rslt[0],eu);
        return rslt;
    }

    static M33f*y_180() {
        M33f*rslt = new M33f[2];
        V3f eu;
        eu(0) = 0;
        eu(1) = 0;
        eu(2) = 0;
        Math::eZYZ_Rmat(rslt[0],eu);
        eu(1) = M_PI;
        Math::eZYZ_Rmat(rslt[1],eu);
        return rslt;
    }

    static M33f*cone_flip() {
        return y_180();
    }

    static M33f*Cn(int N) {
        if( N > 0 ) {
            M33f*rslt = new M33f[N];
            V3f eu;
            eu(0) = 0;
            eu(1) = 0;
            eu(2) = 0;
            for(int n=0;n<N;n++) {
                float angle = 2*M_PI*float(n)/float(N);
                eu(0) = angle;
                Math::eZYZ_Rmat(rslt[n],eu);
            }
            return rslt;
        }
        else
            return NULL;
    }
    
    static M33f*cuboctahedral() {
        M33f*rslt = new M33f[24];
		V3f eu;
		M33f tmp;
		eu(0) = 0;
		eu(1) = 0;
		eu(2) = 0;
		for(int n=0;n<4;n++) {
			float angle = M_PI*float(n)/float(4);
			eu(0) = angle;
			Math::eZYZ_Rmat(rslt[n],eu);
		}
		
		eu(0) = 0;
		eu(1) = M_PI/2;
		eu(2) = 0;
		Math::eZYZ_Rmat(tmp,eu);
		for(int n=0;n<4;n++) {
			rslt[n+ 4] = tmp*rslt[n];
		}
		
		eu(0) = 0;
		eu(1) = -M_PI/2;
		eu(2) = 0;
		Math::eZYZ_Rmat(tmp,eu);
		for(int n=0;n<4;n++) {
			rslt[n+ 8] = tmp*rslt[n];
		}
		
		eu(0) = 0;
		eu(1) = M_PI/2;
		eu(2) = 0;
		Math::eZXZ_Rmat(tmp,eu);
		for(int n=0;n<4;n++) {
			rslt[n+12] = tmp*rslt[n];
		}
		
		eu(0) = 0;
		eu(1) = -M_PI/2;
		eu(2) = 0;
		Math::eZXZ_Rmat(tmp,eu);
		for(int n=0;n<4;n++) {
			rslt[n+16] = tmp*rslt[n];
		}
		
		eu(0) = 0;
		eu(1) = M_PI;
		eu(2) = 0;
		Math::eZYZ_Rmat(tmp,eu);
		for(int n=0;n<4;n++) {
			rslt[n+20] = tmp*rslt[n];
		}
		
		return rslt;
    }
};

#endif /// ANGLES_SYMMETRY_H



