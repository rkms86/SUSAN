#ifndef DATATYPES_H
#define DATATYPES_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

typedef uint8_t  uint8;
typedef int32_t  int32;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef float    single;

#define SUSAN_FILENAME_LENGTH 1024
#define SUSAN_CHUNK_SIZE (1024*1024)
#define SUSAN_FLOAT_TOL 5e-8
#define SUSAN_MAX_N_GPU 16
#define SUSAN_CUDA_THREADS 1024
#define SUSAN_CUDA_WARP 32

#define RAD2DEG 57.295779513082323
#define DEG2RAD  0.017453292519943

#define SUSAN_SVG_SHADOW_BG "#F7F7F7"
//#define SUSAN_SVG_FG_A "#FDB366"
#define SUSAN_SVG_FG_A "#F67E4B"
//#define SUSAN_SVG_FG_B "#98CAE1"
#define SUSAN_SVG_FG_B "#6EA6CD"
//#define SUSAN_SVG_FG_C "#EAECCC"
#define SUSAN_SVG_FG_C "#CFD48A"

typedef struct {
    float x;
    float y;
} Vec2;

typedef struct {
    float x;
    float y;
    float z;
} Vec3;

typedef struct {
    uint32 x;
    uint32 y;
    uint32 z;
} VUInt3;

typedef struct {
    float xx;
    float xy;
    float xz;
    float yx;
    float yy;
    float yz;
    float zx;
    float zy;
    float zz;
} Rot33; /// Row major

typedef struct {
	uint32  ptcl_id;
    uint32  tomo_id;
    uint32  tomo_cix;
    Vec3    pos;      /// Angstroms
    uint32  ref_cix;
    uint32  half_id;
    single  extra_1;
    single  extra_2;
} PtclInf;

typedef struct {
    float U;
    float V;
    float angle;
    float ph_shft;
    float Bfactor;
    float ExpFilt;
    float max_res;
    float score;
} Defocus;

typedef struct {
    float CsLambda3PiH;
    float LambdaPi;
    float AC;
    float CA;
    float apix;
} CtfConst;

typedef struct {
    Rot33 R;
    Vec3  t;
    float w;
} Proj2D;

typedef enum {
	DONE  = -1,
    EMPTY = 0,
    READY = 1
} Status_t;


#endif


