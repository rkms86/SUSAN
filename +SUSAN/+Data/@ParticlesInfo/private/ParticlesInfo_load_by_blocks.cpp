#define MW_NEEDS_VERSION_H /// For Matlab > R2018b
#include "mex.h"
#include "math.h"

#include "matlab.h"
#include "datatypes.h"
#include "math_cpu.h"
#include "particles.h"
#include "io.h"

using namespace Matlab;

#define FUNC_NAME "ParticlesInfo.load_by_blocks"

/// [uint32_block,single_block,ali_block,prj_block,defocus] = load_by_blocks(filename);
void mexFunction(int nOut, mxArray *pOut[], int nIn, const mxArray *pIn[]) {

    ///
    if( nOut != 5 ) {
        mexErrMsgTxt("[" FUNC_NAME "] Five outputs required");
    }

    ///
    if( nIn  != 1 ) {
        mexErrMsgTxt("[" FUNC_NAME "] One input required");
    }

    /// Validate Inputs' types:
    if( !is_char(pIn[0]) )
        mexErrMsgTxt("[" FUNC_NAME "] Input 1 must be a char string.");

    /// Get Inputs:
    mwSize L;
    int    num_ptcl, num_proj, num_refs;

    char *filename;
    get_array(pIn[0],filename,L);

    /// Load
    if( !Particles::check_signature(filename) ) mexErrMsgTxt("[" FUNC_NAME "] Input 1 must be a char string.\n");    
    Particle cur_ptcl;
    ParticlesInStream ptcl_stream(filename);
    ptcl_stream.get(cur_ptcl);
    num_ptcl = ptcl_stream.n_ptcl;
    num_proj = ptcl_stream.n_proj;
    num_refs = ptcl_stream.n_refs;
    
    /// Allocate Output
    uint32_t *p_uint32_block;
    allocate_and_get(p_uint32_block,pOut[0],num_ptcl,5,1);

    float *p_single_block;
    allocate_and_get(p_single_block,pOut[1],num_ptcl,5,1);

    float *p_ali_block;
    allocate_and_get(p_ali_block,pOut[2],num_ptcl,8,num_refs);

    float *p_prj_block;
    allocate_and_get(p_prj_block,pOut[3],num_proj,7,num_ptcl);

    float *p_ctf_block;
    allocate_and_get(p_ctf_block,pOut[4],num_proj,7,num_ptcl);

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
    
    /// Iterate over particles:
    for(int ptcl=0; ptcl<num_ptcl; ptcl++) {
    
        /// General particle info
        if( !ptcl_stream.read_buffer() )
			mexErrMsgTxt("[" FUNC_NAME "] Trying to read more particles than the available\n.");
        
        ptcl_id  [ptcl] = cur_ptcl.ptcl_id();
        tomo_id  [ptcl] = cur_ptcl.tomo_id();
        tomo_cix [ptcl] = cur_ptcl.tomo_cix();
        pos_x    [ptcl] = cur_ptcl.pos().x;
        pos_y    [ptcl] = cur_ptcl.pos().y;
        pos_z    [ptcl] = cur_ptcl.pos().z;
        class_cix[ptcl] = cur_ptcl.ref_cix();
        half_id  [ptcl] = cur_ptcl.half_id();
        xtra1    [ptcl] = cur_ptcl.extra_1();
        xtra2    [ptcl] = cur_ptcl.extra_2();
        
        /// Alignment per reference/class
        for(int refs=0; refs<num_refs; refs++) {
			
            float *cur_ali_info = p_ali_block + refs*7*num_ptcl;
            cur_ali_info[ptcl             ] = cur_ptcl.ali_eu[refs].x*180/M_PI;
            cur_ali_info[ptcl +   num_ptcl] = cur_ptcl.ali_eu[refs].y*180/M_PI;
            cur_ali_info[ptcl + 2*num_ptcl] = cur_ptcl.ali_eu[refs].z*180/M_PI;
            cur_ali_info[ptcl + 3*num_ptcl] = cur_ptcl.ali_t [refs].x;
            cur_ali_info[ptcl + 4*num_ptcl] = cur_ptcl.ali_t [refs].y;
            cur_ali_info[ptcl + 5*num_ptcl] = cur_ptcl.ali_t [refs].z;
            cur_ali_info[ptcl + 6*num_ptcl] = cur_ptcl.ali_cc[refs];
            cur_ali_info[ptcl + 7*num_ptcl] = cur_ptcl.ali_w [refs];
        }
        
		/// 2D refinement
        float *cur_prj_info = p_prj_block + ptcl*7*num_proj;
        for(int proj=0; proj<num_proj; proj++) {
			
            cur_prj_info[proj             ] = cur_ptcl.prj_eu[proj].x*180/M_PI;
            cur_prj_info[proj +   num_proj] = cur_ptcl.prj_eu[proj].y*180/M_PI;
            cur_prj_info[proj + 2*num_proj] = cur_ptcl.prj_eu[proj].z*180/M_PI;
            cur_prj_info[proj + 3*num_proj] = cur_ptcl.prj_t [proj].x;
            cur_prj_info[proj + 4*num_proj] = cur_ptcl.prj_t [proj].y;
            cur_prj_info[proj + 5*num_proj] = cur_ptcl.prj_cc[proj];
            cur_prj_info[proj + 6*num_proj] = cur_ptcl.prj_w [proj];

        }
        
		/// CTF:
        float *cur_ctf_info = p_ctf_block + ptcl*7*num_proj;
        for(int proj=0; proj<num_proj; proj++) {
			
            cur_ctf_info[proj             ] = cur_ptcl.def[proj].U;
            cur_ctf_info[proj +   num_proj] = cur_ptcl.def[proj].V;
            cur_ctf_info[proj + 2*num_proj] = cur_ptcl.def[proj].angle;
            cur_ctf_info[proj + 3*num_proj] = cur_ptcl.def[proj].Bfactor;
            cur_ctf_info[proj + 4*num_proj] = cur_ptcl.def[proj].ExpFilt;
            cur_ctf_info[proj + 5*num_proj] = cur_ptcl.def[proj].max_res;
            cur_ctf_info[proj + 6*num_proj] = cur_ptcl.def[proj].score;
        }
        
    }
}


