#ifndef EM_H
#define EM_H

#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <string>
#include "datatypes.h"

using namespace std;

namespace EM {
    void write(const single *data, const uint32 X, const uint32 Y, const uint32 Z, const char*mapname) {
        FILE*fp=fopen(mapname,"wb");
        uint32 header[128];
        uint8 *p8 = (uint8*)(header);
        for(int i=0;i<128;i++) header[i] = 0;
        p8[0] = 6;
        p8[3] = 5;
        header[1]  = X;
        header[2]  = Y;
        header[3]  = Z;
        fwrite((void*)header,sizeof(uint32),128,fp);
        fwrite((void*)data,sizeof(single),X*Y*Z,fp);
        fclose(fp);
    }

}

#endif 

