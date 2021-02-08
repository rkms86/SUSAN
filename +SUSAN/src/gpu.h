#ifndef GPU_H
#define GPU_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

#include "datatypes.h"

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cufft.h"

namespace GPU {

inline int div_round_up(int num, int den) {
    return  (num + den - 1) / den;
}

dim3 get_block_size_2D() {
    dim3 rslt( SUSAN_CUDA_WARP, div_round_up(SUSAN_CUDA_THREADS,SUSAN_CUDA_WARP), 1 );
    return rslt;
}

dim3 get_block_size_3D(int z) {
    int den = SUSAN_CUDA_THREADS / z;
    dim3 rslt( SUSAN_CUDA_WARP, div_round_up(den,SUSAN_CUDA_WARP), z );
    return rslt;
}

dim3 calc_block_size(int th, int ws, int z) {
    int den = th / z;
    dim3 rslt( ws, div_round_up(den,ws), z );
    return rslt;
}

dim3 calc_grid_size(dim3&block_size, int X, int Y, int Z) {
    dim3 rslt( div_round_up(X,block_size.x), div_round_up(Y,block_size.y), div_round_up(Z,block_size.z) );
    return rslt;
}

int count_devices() {
	int devices=0;
    cudaError_t err = cudaGetDeviceCount(&devices);
    if( err != cudaSuccess ) {
        fprintf(stderr,"Error counting CUDA devices.");
        fprintf(stderr,"GPU error: %s.\n",cudaGetErrorString(err));
        exit(1);
    }
    return devices;
}

void set_device(uint32 device) {
    cudaError_t err = cudaSetDevice(device);
    if( err != cudaSuccess ) {
        fprintf(stderr,"Error accesing CUDA device %d. ",device);
        fprintf(stderr,"GPU error: %s.\n",cudaGetErrorString(err));
        exit(1);
    }
}

void sync() {
    cudaError_t err = cudaDeviceSynchronize();
    if( err != cudaSuccess ) {
        fprintf(stderr,"Error synchronizing CUDA device. ");
        fprintf(stderr,"GPU error: %s.\n",cudaGetErrorString(err));
        exit(1);
    }
}

void reset() {
    cudaError_t err = cudaDeviceReset();
    if( err != cudaSuccess ) {
        fprintf(stderr,"Error resetting CUDA device. ");
        fprintf(stderr,"GPU error: %s.\n",cudaGetErrorString(err));
        exit(1);
    }
}

class Stream {
public:
    cudaStream_t strm;

    Stream() {
        strm = 0;
    }

    void configure() {
        cudaError_t err = cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking);
        if( err != cudaSuccess ) {
            fprintf(stderr,"Error CUDA couldn't create stream. ");
            fprintf(stderr,"GPU error: %s.\n",cudaGetErrorString(err));
            exit(1);
        }
    }

    void sync() {
        cudaError_t err = cudaStreamSynchronize(strm);
        if( err != cudaSuccess ) {
            fprintf(stderr,"Error synchronizing CUDA stream. ");
            fprintf(stderr,"GPU error: %s.\n",cudaGetErrorString(err));
            exit(1);
        }
    }


    ~Stream() {
        if( strm != 0 ) {
            cudaStreamDestroy(strm);
        }
    }
};

template<class T>
class GArr {

public:
    T*ptr;

protected:
	size_t internal_numel;

public:
    GArr() {
        ptr = NULL;
    }

    ~GArr() {
        free();
    }

    void alloc(const size_t numel) {
        free();
        cudaError_t err = cudaMalloc( (void**)(&ptr), sizeof(T)*numel );
        if( err != cudaSuccess ) {
            fprintf(stderr,"Error allocating CUDA memory [%dx%d bytes]. ",sizeof(T),numel);
            fprintf(stderr,"GPU error: %s.\n",cudaGetErrorString(err));
            exit(1);
        }
        internal_numel = numel;
    }
    
    void clear() {
		cudaError_t err = cudaMemset( ptr, 0, sizeof(T)*internal_numel );
        if( err != cudaSuccess ) {
            fprintf(stderr,"Error clearing CUDA memory. ");
            fprintf(stderr,"GPU error: %s.\n",cudaGetErrorString(err));
            exit(1);
        }
	}
	
	void clear(cudaStream_t strm) {
		cudaError_t err = cudaMemsetAsync( ptr, 0, sizeof(T)*internal_numel, strm );
        if( err != cudaSuccess ) {
            fprintf(stderr,"Error clearing CUDA memory (async). ");
            fprintf(stderr,"GPU error: %s.\n",cudaGetErrorString(err));
            exit(1);
        }
	}

protected:
    void free() {
        if( ptr != NULL )
            cudaFree(ptr);
    }
};

typedef GArr<uint32>  GArrUint32;
typedef GArr<single>  GArrSingle;
typedef GArr<double>  GArrDouble;
typedef GArr<float2>  GArrSingle2;
typedef GArr<float3>  GArrSingle3;
typedef GArr<float4>  GArrSingle4;
typedef GArr<double2> GArrDouble2;
typedef GArr<Defocus> GArrDefocus;
typedef GArr<Proj2D>  GArrProj2D;
typedef GArr<Vec3>    GArrVec3;

template<class T>
class GTex2D {

public:
    cudaTextureObject_t texture;
    cudaSurfaceObject_t surface;

protected:

    cudaArray*              g_arr;
    cudaChannelFormatDesc   chn_desc;
    struct cudaResourceDesc res_desc;
    struct cudaTextureDesc  tex_desc;

public:
    GTex2D() {
        texture = 0;
        surface = 0;
        g_arr   = NULL;
    }

    ~GTex2D() {
        free();
    }

    void alloc(const uint32 x, const uint32 y, const uint32 z) {

        cudaError_t err;
        cudaExtent vol = {x,y,z};
        chn_desc = cudaCreateChannelDesc<T>();
        err = cudaMalloc3DArray(&g_arr, &chn_desc, vol, cudaArraySurfaceLoadStore | cudaArrayLayered);
        if( err != cudaSuccess ) {
            fprintf(stderr,"Error allocating CUDA 3D array. ");
            fprintf(stderr,"GPU error: %s.\n",cudaGetErrorString(err));
            exit(1);
        }

        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = g_arr;

        memset(&tex_desc, 0, sizeof(tex_desc));
        tex_desc.addressMode[0]   = cudaAddressModeBorder;
        tex_desc.addressMode[1]   = cudaAddressModeBorder;
        tex_desc.filterMode       = cudaFilterModeLinear;
        tex_desc.readMode         = cudaReadModeElementType;
        tex_desc.normalizedCoords = 0;

        err = cudaCreateTextureObject(&texture, &res_desc, &tex_desc, NULL);
        if( err != cudaSuccess ) {
            fprintf(stderr,"Error creating CUDA texture object. ");
            fprintf(stderr,"GPU error: %s.\n",cudaGetErrorString(err));
            exit(1);
        }

        err = cudaCreateSurfaceObject(&surface, &res_desc);
        if( err != cudaSuccess ) {
            fprintf(stderr,"Error creating CUDA surface object. ");
            fprintf(stderr,"GPU error: %s.\n",cudaGetErrorString(err));
            exit(1);
        }
    }

protected:
    void free() {
        if( g_arr   != NULL ) cudaFreeArray(g_arr);
        if( texture != 0    ) cudaDestroyTextureObject(texture);
        if( surface != 0    ) cudaDestroySurfaceObject(surface);
    }

};

typedef GTex2D<single> GTex2DSingle;
typedef GTex2D<float2> GTex2DSingle2;

template<class T>
class GHost {

public:
    T*ptr;

public:
    GHost() {
        ptr = NULL;
    }

    ~GHost() {
        free();
    }

    void alloc(const size_t numel) {
        free();
        cudaError_t err = cudaMallocHost( (void**)(&ptr), sizeof(T)*numel );
        if( err != cudaSuccess ) {
            fprintf(stderr,"Error allocating CUDA-host memory [%dx%d bytes]. ",sizeof(T),numel);
            fprintf(stderr,"GPU error: %s.\n",cudaGetErrorString(err));
            exit(1);
        }
    }

protected:
    void free() {
        if( ptr != NULL )
            cudaFreeHost(ptr);
    }
};

typedef GHost<single>  GHostSingle;
typedef GHost<double>  GHostDouble;
typedef GHost<float2>  GHostFloat2;
typedef GHost<Proj2D>  GHostProj2D;
typedef GHost<double2> GHostDouble2;
typedef GHost<Defocus> GHostDefocus;

template<class T>
void upload_async(T*p_gpu,const T*p_cpu, size_t numel, cudaStream_t&strm) {
    cudaError_t err = cudaMemcpyAsync( (void*)(p_gpu), (const void*)p_cpu, sizeof(T)*numel , cudaMemcpyHostToDevice, strm);
    if( err != cudaSuccess ) {
        fprintf(stderr,"Error uploading async to CUDA memory. ");
        fprintf(stderr,"GPU error: %s.\n",cudaGetErrorString(err));
        exit(1);
    }
}

template void upload_async<>(single* ,const single* ,size_t,cudaStream_t&);
template void upload_async<>(float2* ,const float2* ,size_t,cudaStream_t&);
template void upload_async<>(Proj2D* ,const Proj2D* ,size_t,cudaStream_t&);
template void upload_async<>(Defocus*,const Defocus*,size_t,cudaStream_t&);

template<class T>
void download_async(T*p_cpu,const T*p_gpu, size_t numel, cudaStream_t&strm) {
    cudaError_t err = cudaMemcpyAsync( (void*)(p_cpu), (const void*)p_gpu, sizeof(T)*numel , cudaMemcpyDeviceToHost, strm);
    if( err != cudaSuccess ) {
        fprintf(stderr,"Error downloading async to CUDA memory. ");
        fprintf(stderr,"GPU error: %s.\n",cudaGetErrorString(err));
        exit(1);
    }
}

template void download_async<>(single* ,const single* ,size_t,cudaStream_t&);

}

#endif /// GPU_H
