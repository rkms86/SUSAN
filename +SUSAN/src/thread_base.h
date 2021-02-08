#ifndef THREAD_BASE_H
#define THREAD_BASE_H

#include <pthread.h>
#include "datatypes.h"
#include "thread_sharing.h"


class PThread {

public:
	pthread_t pth_id;
	
	PThread() {}
	
    bool start() {
        return (pthread_create(&pth_id,NULL,PThread::exec_main,(void*)this) == 0);
    }

    bool wait() {
        return (pthread_join(pth_id,NULL) == 0);
    }
	
protected:
	virtual void main() = 0;
	
private:
    static void *exec_main(void*ctx) {
        PThread *th = (PThread*)ctx;
        th->main();
        return NULL;
    }	
	
};

class Worker : public PThread {
	
public:
	uint32 worker_id;
	uint32 work_progress;
	uint32 work_accumul;
	WorkerCommand *worker_cmd;
	
	Worker() {
		worker_id     = 0;
		work_progress = 0;
		work_accumul  = 0;
		worker_cmd    = NULL;
	}
	
	
	
};

#endif /// THREAD_BASE_H

