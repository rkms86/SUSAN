#ifndef POOL_COORDINATOR_H
#define POOL_COORDINATOR_H

#include <pthread.h>
#include "datatypes.h"
#include "memory.h"
#include "thread_sharing.h"
#include "thread_base.h"
#include "particles.h"
#include "tomogram.h"
#include "stack_reader.h"

class PoolCoordinator : public PThread {

public:
	DoubleBufferHandler *double_buffer;
	Tomograms   *tomos;
	int num_threads;
	
	PoolCoordinator(StackReader&stkrdr,int in_num_threads) {
		double_buffer = stkrdr.double_buffer;
		tomos = stkrdr.tomos;
		num_threads = in_num_threads;
	}	

protected:
	void main() {
		coord_init();
		while( double_buffer->RO_get_status() > DONE ) {
			if( double_buffer->RO_get_status() == READY ) {
				StackBuffer*ptr = (StackBuffer*)double_buffer->RO_get_buffer();
				coord_main(ptr->stack,ptr->ptcls,tomos->at(ptr->tomo_ix));
			}
			double_buffer->RO_sync();
		}
		coord_end();
	}
	
	virtual void coord_init() {
	}
	
	virtual void coord_main(float*stack,ParticlesSubset&ptcls,Tomogram&tomo) {
		printf(" Processing TomoID %d: %d particles\n",tomo.tomo_id,ptcls.n_ptcl);
		Particle ptcl;
		ptcls.get(ptcl,0);
		printf("   First Particle ID: %d [%d]\n",ptcl.ptcl_id(),ptcl.tomo_id());
		ptcls.get(ptcl,ptcls.n_ptcl-1);
		printf("   Last Particle ID: %d [%d]\n",ptcl.ptcl_id(),ptcl.tomo_id());
	}
	
	virtual void coord_end() {
	}
	
};


#endif /// POOL_COORDINATOR_H

