/*
 * This file is part of the Substack Analysis (SUSAN) framework.
 * Copyright (c) 2018-2022 Ricardo Miguel Sanchez Loayza.
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

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

typedef uint8_t  uint8;
typedef int32_t  int32;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef float    single;

#define SUSAN_CARRIER_RETURN '\r'
//#define SUSAN_CARRIER_RETURN 0

#define SUSAN_FILENAME_LENGTH 1024
#define SUSAN_CHUNK_SIZE (1024*1024)
#define SUSAN_FLOAT_TOL 5e-8
#define SUSAN_MAX_N_GPU 24
#define SUSAN_CUDA_THREADS 1024
#define SUSAN_CUDA_WARP 32

#define RAD2DEG 57.295779513082323
#define DEG2RAD  0.017453292519943

#define SUSAN_SVG_SHADOW_BG "#E6E6E6"
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
    float angle;   /// Sexagesimal degrees
    float ph_shft; /// Radians
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

typedef enum {
    NO_NORM=0,
    ZERO_MEAN,
    ZERO_MEAN_W_STD,
    ZERO_MEAN_1_STD,
    GAT_RAW,
    GAT_NORMAL
} NormalizationType_t;

typedef enum {
    PAD_ZERO=0,
    PAD_GAUSSIAN
} PaddingType_t;

typedef enum {
    ALI_NO_INV=0,
    ALI_ON_REFERENCE,
    ALI_ON_SUBSTACK,
    ALI_ON_SUBSTACK_SSNR,
    ALI_CUMULATIVE_FSC
} CtfAlignmentType_t;

typedef enum {
    CC_NONE=0,
    CC_PROB,
    CC_SIGMA
} CcStatsType_t;

typedef enum {
    INV_NO_INV=0,
    INV_PHASE_FLIP,
    INV_WIENER,
    INV_WIENER_SSNR
} CtfInversionType_t;

typedef enum {
    ELLIPSOID,
    CYLINDER,
    CIRCLE,
    CUBOID,
} OffsetType_t;



