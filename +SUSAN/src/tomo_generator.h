#ifndef TOMO_GENERATOR_H
#define TOMO_GENERATOR_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "datatypes.h"
#include "memory.h"
#include "io.h"
#include "math_cpu.h"
#include "mrc.h"

#include "particles.h"
#include "tomogram.h"

class TomoParticles : public Particles {
public:
    TomoParticles(uint32 box_size,Tomograms&tomos) {
        n_proj = tomos.num_proj;
		n_refs = 1;
		n_ptcl = get_num_ptcls(box_size,tomos);
		allocate(n_ptcl,n_proj,n_refs);
		populate_tomos(box_size,tomos);
    }

    ~TomoParticles() {
		free(p_raw);
    }

protected:
	uint32 get_num_ptcls(uint32 box_size,Tomograms&tomos) {
		uint32 n = 0;
		for(int i=0;i<tomos.num_tomo;i++) {
			int cur_n=1;
			cur_n *= (2*get_n( tomos[i].tomo_dim.x, box_size ) + 1);
			cur_n *= (2*get_n( tomos[i].tomo_dim.y, box_size ) + 1);
			cur_n *= get_z( tomos[i].tomo_dim.z, box_size );
			n += cur_n;
		}
		return n;
	}
	
	int get_n(uint32 l,uint32 n) {
		float data = (l-n)/2;
		return (int)floor( data/n );
	}
	
	int get_z(uint32 l,uint32 n) {
		float data = l;
		return (int)ceil( data/n );
	}
	
	void populate_tomos(uint32 box_size,Tomograms&tomos) {
		int ix = 0;
		Particle ptcl;
		for(int t=0;t<tomos.num_tomo;t++) {
			
			int n_z = get_z(tomos[t].tomo_dim.z,box_size);
			
			for(int z=0;z<n_z;z++) {
				
				float z_out = (float)(z*box_size+box_size/2);
				z_out = z_out - tomos[t].tomo_center(2);
				
				int n_y = get_n(tomos[t].tomo_dim.y,box_size);
			
				for(int y=-n_y;y<=n_y;y++) {
					
					int n_x = get_n(tomos[t].tomo_dim.x,box_size);
			
					for(int x=-n_x;x<=n_x;x++) {
						
						if( !get(ptcl,ix) ) {
							fprintf(stderr,"Error creating particles info for the tomograms creation\n");
							exit(1);
						}
						
						ptcl.ptcl_id()  = ix;
						ptcl.tomo_cix() = t;
						ptcl.tomo_id()  = tomos[t].tomo_id;
						ptcl.pos().x    = ((float)x*box_size)*tomos[t].pix_size;
						ptcl.pos().y    = ((float)y*box_size)*tomos[t].pix_size;
						ptcl.pos().z    = z_out*tomos[t].pix_size;
						
						//for(int k=0;k<tomos[t].num_proj;k++) {
						for(int k=30;k<31;k++) {
							ptcl.prj_w[k] = tomos[t].w[k];
						}
						
						ix++;
					} // X
					
				} // Y
				
			} // Z
			
		} // tomos
	}

};

class TomoRec {
public:
	int current_z;
	
	int N;
	int X;
	int Y;
	int Z;
	int XY;
	
	float apix;
	
	float*buffer;
	
	FILE*fp;
	
	TomoRec(int box_size,int numel) {
		current_z = -1;
		N = box_size;
		buffer = new float[N*numel];
	}
	
	~TomoRec() {
		delete [] buffer;
	}
	
	bool start_rec(const char*filename,Tomogram*tomo) {
		Mrc::Header_t hdr;
		memset((void*)(&hdr),0,sizeof(Mrc::Header_t));
		hdr.datax = tomo->tomo_dim.x;
		hdr.datay = tomo->tomo_dim.y;
		hdr.dataz = tomo->tomo_dim.z;
		hdr.mode  = 2;
		hdr.gridx = tomo->tomo_dim.x;
		hdr.gridy = tomo->tomo_dim.y;
		hdr.gridz = tomo->tomo_dim.z;
		hdr.xlen  = tomo->pix_size*(float)tomo->tomo_dim.x;
		hdr.ylen  = tomo->pix_size*(float)tomo->tomo_dim.y;
		hdr.zlen  = tomo->pix_size*(float)tomo->tomo_dim.z;
		hdr.mapc = 1;
		hdr.mapr = 2;
		hdr.maps = 3;
		fp = fopen(filename,"wb");
		if( fp == NULL ) {
			fprintf(stderr,"Error creating %s file.\n",filename);
			exit(1);
		}
		fwrite((void*)(&hdr),sizeof(Mrc::Header_t),1,fp);
		
		X = tomo->tomo_dim.x;
		Y = tomo->tomo_dim.y;
		Z = tomo->tomo_dim.z;
		XY = X*Y;
		
		current_z = -1;
		return true;
	}
	
	void add_block(single*map,const V3f&pt) {
		int in_x = round(pt(0))-N/2;
		int in_y = round(pt(1))-N/2;
		int in_z = round(pt(2));
		if( current_z < 0 ) {
			current_z = in_z;
		}
		else {
			if( in_z > current_z ) {
				commit_buffer();
				current_z = in_z;
			}
		}
		
		float*p_in  = map;
		float*p_out = buffer + in_x + in_y*X;
		
		for(int z=0;z<N;z++) {
			for(int y=0;y<N;y++) {
				memcpy(p_out+y*X,p_in+y*N,sizeof(float)*N);
			}
			p_in  += N*N;
			p_out += XY;
		}
	}
	
	void end_rec() {
		commit_buffer();
		fclose(fp);
	}
	
protected:
	void commit_buffer() {
		int z = current_z-N/2;
		z = min(Z-z,N);
		fwrite((void*)(buffer),sizeof(single),z*XY,fp);
	}
	
};

#endif /// TOMO_GENERATOR_H


