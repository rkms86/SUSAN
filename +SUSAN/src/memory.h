#ifndef MEMORY_H
#define MEMORY_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

template<class T>
inline void free_array(T* &ptr) {
    if( ptr != NULL ) {
        delete [] ptr;
        ptr = NULL;
    }
}

#endif


