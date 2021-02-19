#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <unistd.h>

#include "io.h"
#include "reconstruct.h"
#include "particles.h"
#include "tomogram.h"
#include "reconstruct_args.h"
#include "mpi_iface.h"

class RecPoolMpi : public RecPool {

public:
    MpiInterface*mpi_iface;
    MpiProgress *mpi_progress;

    RecPoolMpi(ArgsRec::Info*info,int n_refs,int in_max_K,int num_ptcls,StackReader&stkrdr,int in_num_threads)
        : RecPool(info,n_refs,in_max_K,num_ptcls,stkrdr,in_num_threads)
    {
    }

    void set_mpi(MpiInterface*in_mpi_iface,MpiProgress*in_mpi_progress,Particles*ptcls_full) {
        mpi_iface = in_mpi_iface;
        mpi_progress = in_mpi_progress;
        unsigned int val = 0;
        mpi_progress->put( mpi_iface->node_id, val );

        if( mpi_iface->is_main_node() ) {
            n_ptcls = ptcls_full->n_ptcl;
        }
    }

protected:
    unsigned int count_progress_mpi() {
        unsigned int rslt = 0;
        if( mpi_iface->is_main_node() ) {
            for(unsigned int i=0;i<mpi_iface->num_nodes;i++) {
                rslt += mpi_progress->get(i);
            }
        }
        return rslt;
    }

    void progress_start() {
        if( mpi_iface->is_main_node() ) {
            RecPool::progress_start();
        }
    }

    void show_progress(const int ptcls_in_tomo) {

        while( (count_progress()) < ptcls_in_tomo ) {

            unsigned int local_progress = count_accumul();
            mpi_progress->put( mpi_iface->node_id, local_progress );

            if( mpi_iface->is_main_node() ) {
                unsigned int total_progress = count_progress_mpi();
                if( total_progress > 0 ) {
                    memset(progress_buffer,' ',66);
                    float progress_percent = 100*(float)total_progress/float(n_ptcls);
                    sprintf(progress_buffer,"        Filling fourier space: %6.2f%%%%",progress_percent);
                    int n = strlen(progress_buffer);
                    add_etc(progress_buffer+n,total_progress,n_ptcls);
                    printf(progress_clear);
                    fflush(stdout);
                    printf(progress_buffer);
                    fflush(stdout);
                }
            }

            sleep(2);
        } /// while
    }

    void show_done() {
        if( mpi_iface->is_main_node() ) {
            memset(progress_buffer,' ',66);
            sprintf(progress_buffer,"        Filling fourier space: 100.00%%%%");
            int n = strlen(progress_buffer);
            progress_buffer[n] = ' ';
            printf(progress_clear);
            printf(progress_buffer);
            printf("\n");
            fflush(stdout);
        }
    }

    void reconstruct_results() {
        double *tmp = new double[2*MP*NP*NP];

        for(int r=0;r<R;r++) {
            MPI_Reduce(workers[0].c_acc[r],tmp,2*MP*NP*NP,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
            if( mpi_iface->is_main_node() ) {
                memcpy(workers[0].c_acc[r],tmp,sizeof(double)*2*MP*NP*NP);
            }

            MPI_Reduce(workers[0].c_wgt[r],tmp,MP*NP*NP,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
            if( mpi_iface->is_main_node() ) {
                memcpy(workers[0].c_wgt[r],tmp,sizeof(double)*MP*NP*NP);
            }
        }
        delete [] tmp;

        if( mpi_iface->is_main_node() ) {
            RecPool::reconstruct_results();
        }
    }

};


void print_data_info(Particles*ptcls,Tomograms&tomos) {
        printf("\t\tAvailable particles:  %d.\n",ptcls->n_ptcl);
        printf("\t\tNumber of classes:    %d.\n",ptcls->n_refs);
	printf("\t\tTomograms available:  %d.\n",tomos.num_tomo);
	printf("\t\tAvailabe projections: %d (max).\n",tomos.num_proj);
}

int main(int ac, char** av) {

    MpiInterface mpi_iface;
    ArgsRec::Info info;

    if( ArgsRec::parse_args(info,ac,av) ) {

        if( mpi_iface.is_main_node() ) {
            ArgsRec::print(info);
        }

        PBarrier barrier(2);
        ParticlesRW*ptcls_io;
        MpiScatterInfo scatter_info(mpi_iface.num_nodes);
        MpiProgress mpi_progress(mpi_iface.num_nodes);

        if( mpi_iface.is_main_node() )
            printf("\tLoading and scattering data files..."); fflush(stdout);

        Tomograms tomos(info.tomos_in);

        if( mpi_iface.is_main_node() ) {
            ptcls_io = new ParticlesRW(info.ptcls_in);
        }

        Particles*ptcls = mpi_iface.scatter_particles(scatter_info,ptcls_io);

        if( mpi_iface.is_main_node() )
            printf(" Done\n"); fflush(stdout);

        if( mpi_iface.is_main_node() ) {
            print_data_info(ptcls_io,tomos);
        }

        StackReader stkrdr(ptcls,&tomos,&barrier);
        RecPoolMpi pool(&info,ptcls->n_refs,tomos.num_proj,ptcls->n_ptcl,stkrdr,info.n_threads);
        pool.set_mpi(&mpi_iface,&mpi_progress,ptcls_io);

        stkrdr.start();
        pool.start();

        stkrdr.wait();
        pool.wait();

        mpi_iface.delete_scattered_ptcls(ptcls);

        if( mpi_iface.is_main_node() ) {
            delete ptcls_io;
        }
    }
    else {
        fprintf(stderr,"Error parsing input arguments.\n");
        exit(1);
    }

    return 0;
}



