#define MW_NEEDS_VERSION_H /// For Matlab > R2018b
#include "mex.h"
#include "math.h"

#include "matlab.h"
#include "datatypes.h"
#include "math_cpu.h"
#include "particles.h"

using namespace Matlab;

#define FUNC_NAME "ParticlesInfo.save_by_blocks"

/// save_by_blocks(filename,uint32_block,single_block,ali_block,prj_block,defocus);
void mexFunction(int nOut, mxArray *pOut[], int nIn, const mxArray *pIn[]) {

    ///
    if( nOut != 0 ) {
        mexErrMsgTxt("[" FUNC_NAME "] No output required");
    }

    ///
    if( nIn  != 6 ) {
        mexErrMsgTxt("[" FUNC_NAME "] Six inputs required");
    }

    /// Validate Inputs' types:
    if( !is_char(pIn[0]) )
        mexErrMsgTxt("[" FUNC_NAME "] Input 1 must be a char string.");

    if( !is_uint32(pIn[1]) )
        mexErrMsgTxt("[" FUNC_NAME "] Input 2 must be a uint32 matrix.");

    for(int i=2;i<6;i++) {
        if( !is_single(pIn[i]) ) {
            char tmp[1024];
            sprintf(tmp,"[" FUNC_NAME "] Input %d must be a single matrix",i+1);
            mexErrMsgTxt(tmp);
        }
    }

    /// Get Inputs:
    mwSize X,Y,Z,L;
    int    num_ptcl, num_proj, num_refs;

    char *filename;
    get_array(pIn[0],filename,L);

    uint32_t *p_uint32_block;
    get_array(pIn[1],p_uint32_block,X,Y,Z);
    num_ptcl = X;
    if( Y != 5 || Z != 1 ) {
        char tmp[1024];
        sprintf(tmp,"[" FUNC_NAME "] Arg2 wrong size [%d,%d,%d], should be [%d 5 1]",X,Y,Z,num_ptcl);
        mexErrMsgTxt(tmp);
    }

    float *p_single_block;
    get_array(pIn[2],p_single_block,X,Y,Z);
    if( X != num_ptcl || Y != 5 || Z != 1 ) {
        char tmp[1024];
        sprintf(tmp,"[" FUNC_NAME "] Arg3 wrong size [%d,%d,%d], should be [%d 5 1]",X,Y,Z,num_ptcl);
        mexErrMsgTxt(tmp);
    }

    float *p_ali_block;
    get_array(pIn[3],p_ali_block,X,Y,Z);
    num_refs = Z;
    if( X != num_ptcl || Y != 8 ) {
        char tmp[1024];
        sprintf(tmp,"[" FUNC_NAME "] Arg4 wrong size [%d,%d,%d], should be [%d 8 %d]",X,Y,Z,num_ptcl,num_refs);
        mexErrMsgTxt(tmp);
    }

    float *p_prj_block;
    get_array(pIn[4],p_prj_block,X,Y,Z);
    num_proj = X;
    if( Y != 7 || Z != num_ptcl ) {
        char tmp[1024];
        sprintf(tmp,"[" FUNC_NAME "] Arg5 wrong size [%d,%d,%d], should be [%d 7 %d]",X,Y,Z,num_proj,num_ptcl);
        mexErrMsgTxt(tmp);
    }

    float *p_ctf_block;
    get_array(pIn[5],p_ctf_block,X,Y,Z);
    if( X != num_proj || Y != 8 || Z != num_ptcl ) {
        char tmp[1024];
        sprintf(tmp,"[" FUNC_NAME "] Arg6 wrong size [%d,%d,%d], should be [%d 8 %d]",X,Y,Z,num_proj,num_ptcl);
        mexErrMsgTxt(tmp);
    }

    /// Unpack UINT32
    uint32_t *ptcl_id   = p_uint32_block;
    uint32_t *tomo_id   = p_uint32_block +   num_ptcl;
    uint32_t *tomo_cix  = p_uint32_block + 2*num_ptcl;
    uint32_t *class_cix = p_uint32_block + 3*num_ptcl;
    uint32_t *half_id   = p_uint32_block + 4*num_ptcl;

	/// Unpack SINGLE
    float *pos_x = p_single_block;
    float *pos_y = p_single_block +   num_ptcl;
    float *pos_z = p_single_block + 2*num_ptcl;
    float *xtra1 = p_single_block + 3*num_ptcl;
    float *xtra2 = p_single_block + 4*num_ptcl;
    
    Particle cur_ptcl;
    ParticlesOutStream ptcl_stream(filename,num_proj,num_refs);
    ptcl_stream.get(cur_ptcl);
    
    /// Iterate over particles:
    for(int ptcl=0; ptcl<num_ptcl; ptcl++) {
		
		/// General particle info
        cur_ptcl.ptcl_id()  = ptcl_id[ptcl];
        cur_ptcl.tomo_id()  = tomo_id[ptcl];
        cur_ptcl.tomo_cix() = tomo_cix[ptcl];
        cur_ptcl.pos().x    = pos_x[ptcl];
        cur_ptcl.pos().y    = pos_y[ptcl];
        cur_ptcl.pos().z    = pos_z[ptcl];
        cur_ptcl.ref_cix()  = class_cix[ptcl];
        cur_ptcl.half_id()  = half_id[ptcl];
        cur_ptcl.extra_1()  = xtra1[ptcl];
        cur_ptcl.extra_2()  = xtra2[ptcl];
        
        /// Alignment per reference/class
        for(int refs=0; refs<num_refs; refs++) {
            float *cur_ali_info = p_ali_block + refs*8*num_ptcl;
            cur_ptcl.ali_eu[refs].x = cur_ali_info[ptcl             ]*DEG2RAD;
            cur_ptcl.ali_eu[refs].y = cur_ali_info[ptcl +   num_ptcl]*DEG2RAD;
            cur_ptcl.ali_eu[refs].z = cur_ali_info[ptcl + 2*num_ptcl]*DEG2RAD;
            cur_ptcl.ali_t [refs].x = cur_ali_info[ptcl + 3*num_ptcl];
            cur_ptcl.ali_t [refs].y = cur_ali_info[ptcl + 4*num_ptcl];
            cur_ptcl.ali_t [refs].z = cur_ali_info[ptcl + 5*num_ptcl];
            cur_ptcl.ali_cc[refs]   = cur_ali_info[ptcl + 6*num_ptcl];
            cur_ptcl.ali_w [refs]   = cur_ali_info[ptcl + 7*num_ptcl];
        }

		/// 2D refinement
        float *cur_prj_info = p_prj_block + ptcl*7*num_proj;
        for(int proj=0; proj<num_proj; proj++) {
			cur_ptcl.prj_eu[proj].x = cur_prj_info[proj             ]*DEG2RAD;
            cur_ptcl.prj_eu[proj].y = cur_prj_info[proj +   num_proj]*DEG2RAD;
            cur_ptcl.prj_eu[proj].z = cur_prj_info[proj + 2*num_proj]*DEG2RAD;
            cur_ptcl.prj_t [proj].x = cur_prj_info[proj + 3*num_proj];
            cur_ptcl.prj_t [proj].y = cur_prj_info[proj + 4*num_proj];
            cur_ptcl.prj_cc[proj]   = cur_prj_info[proj + 5*num_proj];
            cur_ptcl.prj_w [proj]   = cur_prj_info[proj + 6*num_proj];
        }
        
        
        /// CTF:
        float *cur_ctf_info = p_ctf_block + ptcl*8*num_proj;
        for(int proj=0; proj<num_proj; proj++) {
			cur_ptcl.def[proj].U       = cur_ctf_info[proj             ];
            cur_ptcl.def[proj].V       = cur_ctf_info[proj +   num_proj];
            cur_ptcl.def[proj].angle   = cur_ctf_info[proj + 2*num_proj];
            cur_ptcl.def[proj].ph_shft = cur_ctf_info[proj + 3*num_proj];
            cur_ptcl.def[proj].Bfactor = cur_ctf_info[proj + 4*num_proj];
            cur_ptcl.def[proj].ExpFilt = cur_ctf_info[proj + 5*num_proj];
            cur_ptcl.def[proj].max_res = cur_ctf_info[proj + 6*num_proj];
            cur_ptcl.def[proj].score   = cur_ctf_info[proj + 7*num_proj];
        }
        
        ptcl_stream.write_buffer();
        //cur_ptcl.print();
	}
	
}

