#ifndef MATLAB_H
#define MATLAB_H

/// add -lut -lmwservices when compiling with mex

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "datatypes.h"

#include "mex.h"

/// CHECK COMPLEX COMPATIBILITY
#ifndef MX_HAS_INTERLEAVED_COMPLEX
    #error Interleaved Complex API is disabled. It must be enabled (MX_HAS_INTERLEAVED_COMPLEX is not set).
#endif

namespace Matlab {

    bool is_complex(const mxArray*in_arr) {
        return mxIsComplex(in_arr);
    }

    bool is_int32(const mxArray*in_arr) {
        return ( mxGetClassID(in_arr) == mxINT32_CLASS );
    }

    bool is_uint32(const mxArray*in_arr) {
        return ( mxGetClassID(in_arr) == mxUINT32_CLASS );
    }

    bool is_uint64(const mxArray*in_arr) {
        return ( mxGetClassID(in_arr) == mxUINT64_CLASS );
    }

    bool is_single(const mxArray*in_arr) {
        return ( mxGetClassID(in_arr) == mxSINGLE_CLASS );
    }

    bool is_double(const mxArray*in_arr) {
        return ( mxGetClassID(in_arr) == mxDOUBLE_CLASS );
    }

    bool is_char(const mxArray*in_arr) {
        return ( mxGetClassID(in_arr) == mxCHAR_CLASS );
    }

    template<class T>
    void get_array(const mxArray*in_arr, T* &ptr) {
        ptr = (T*)mxGetData(in_arr);
    }

    template<class T>
    void get_array(const mxArray*in_arr, T* &ptr, mwSize&X, mwSize&Y, mwSize&Z) {
        mwSize ndim = mxGetNumberOfDimensions(in_arr);
        const mwSize *dims = mxGetDimensions(in_arr);
        X = dims[0];
        Y = dims[1];
        if( ndim > 2 )
            Z = dims[2];
        else
            Z = 1;
        ptr = (T*)mxGetData(in_arr);
        return;
    }

    void get_array(const mxArray*in_arr, char* &ptr, mwSize&L) {
        mwSize ndim = mxGetNumberOfDimensions(in_arr);
        const mwSize *dims = mxGetDimensions(in_arr);
        L = (dims[0]>dims[1])?dims[0]:dims[1];
        ptr = mxArrayToString(in_arr);
        return;
    }

    float get_scalar_single(const mxArray*in_arr) {
        float *tmp,rslt;
        get_array(in_arr,tmp);
        rslt = tmp[0];
        return rslt;
    }

    uint32 get_scalar_uint32(const mxArray*in_arr) {
        uint32 *tmp,rslt;
        get_array(in_arr,tmp);
        rslt = tmp[0];
        return rslt;
    }

    void allocate(mxArray* &out_arr, mwSize inX, mwSize inY, mwSize inZ, mxClassID classid, mxComplexity ComplexFlag) {
        mwSize ndim = 3;
        mwSize dims[3];
        dims[0] = inX;
        dims[1] = inY;
        dims[2] = inZ;
        out_arr = mxCreateNumericArray(ndim, dims, classid, ComplexFlag);
    }

    void allocate_and_get(float* &ptr, mxArray* &out_arr, mwSize inX, mwSize inY, mwSize inZ) {
        allocate(out_arr, inX, inY, inZ, mxSINGLE_CLASS, mxREAL);
        get_array(out_arr,ptr);
    }

    void allocate_and_get(uint32* &ptr, mxArray* &out_arr, mwSize inX, mwSize inY, mwSize inZ) {
        allocate(out_arr, inX, inY, inZ, mxUINT32_CLASS, mxREAL);
        get_array(out_arr,ptr);
    }

    void allocate_and_get(double* &ptr, mxArray* &out_arr, mwSize inX, mwSize inY, mwSize inZ) {
        allocate(out_arr, inX, inY, inZ, mxDOUBLE_CLASS, mxREAL);
        get_array(out_arr,ptr);
    }

}


#endif


