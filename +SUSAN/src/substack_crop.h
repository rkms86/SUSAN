#ifndef SUBSTACK_CROP_H
#define SUBSTACK_CROP_H

#include <pthread.h>
#include "datatypes.h"
#include "thread_sharing.h"
#include "thread_base.h"
#include "particles.h"
#include "tomogram.h"
#include "mrc.h"

class SubstackCrop {
	
public:
	Tomogram *tomo;
	bool fill_with_randn;
	float x_min;
	float x_max;
	float y_min;
	float y_max;
	int   N;
	
	SubstackCrop() {
		tomo = NULL;
		fill_with_randn = false;
		N = 200;
	}
	
	void setup(Tomogram*tomos_in,int box_size,bool fill_with_randn_in = false) {
		tomo = tomos_in;
		fill_with_randn = fill_with_randn_in;
		N = box_size;
		float Nh = box_size/2;
		x_min = Nh;
		x_max = float(tomo->stk_dim.x)-Nh;
		y_min = Nh;
		y_max = float(tomo->stk_dim.y)-Nh;
	}
	
	~SubstackCrop() {
	}
	
	bool project_point(V3f&pt_out,const V3f&p_tomo,const int k) {
		if( tomo == NULL )
			return false;
		pt_out = (tomo->R[k]*p_tomo) + tomo->t[k];
		pt_out = (pt_out/tomo->pix_size) + tomo->stk_center;
		return check_point(pt_out);
	}
	
	void get_subpix_shift(Vec3&p_subpix,const V3f&p_proj) {
		p_subpix.x = p_proj(0) - floor(p_proj(0));
		p_subpix.y = p_proj(1) - floor(p_proj(1));
	}
	
	void crop(float*substack,float*stack,const V3f&p_proj,const int k) {
		internal_crop( substack+k*N*N, stack+k*tomo->stk_dim.x*tomo->stk_dim.y, floor(p_proj(0)), floor(p_proj(1)) );
	}
	
	void normalize(float*substack,const int k) {
		internal_normalize( substack+k*N*N );
	}
	
	
protected:
	bool check_point(const V3f&p_proj) {
		return ( p_proj(0) >= x_min && p_proj(0) <= x_max && p_proj(1) >= y_min && p_proj(1) <= y_max );
	}
	
	void internal_crop(float*substack,float*stack,const int x,const int y) {
		int w = tomo->stk_dim.x;
		float *out = substack;
		float *in = stack + x-(N/2) + (y-(N/2))*w;
		for(int j=0;j<N;j++) {
			memcpy((void*)out,(void*)in,N*sizeof(float));
			out += N;
			in  += w;
		}
	}
	
	void internal_normalize(float*substack) {
		float avg,std;
		Math::get_avg_std(avg,std,substack,N*N);
		Math::normalize(substack,N*N,avg,std,1);
	}

};


#endif /// SUBSTACK_CROP_H

