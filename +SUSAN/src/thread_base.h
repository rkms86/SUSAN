/*
 * This file is part of the Substack Analysis (SUSAN) framework.
 * Copyright (c) 2018-2021 Ricardo Miguel Sanchez Loayza.
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

