#ifndef MRC_H
#define MRC_H

#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <string>
#include "datatypes.h"

using namespace std;

namespace Mrc {
	
	typedef struct {
		uint32 datax;
		uint32 datay;
		uint32 dataz;
		uint32 mode;
		int32  xstart;
		int32  ystart;
		int32  zstart;
		uint32 gridx;
		uint32 gridy;
		uint32 gridz;
		single xlen; // Cell size; pixel spacing = nlen/gridN
		single ylen;
		single zlen;
		single alpha;
		single beta;
		single gamma;
		uint32 mapc; // 1,2,3
		uint32 mapr;
		uint32 maps;
		single min;
		single max;
		single avg;
		int32  ispg;
		uint32 nsymbt;
		uint32 pad[(1024-96)/4];
	} Header_t;
  
    float get_apix (const char*mapname) {
        FILE*fp=fopen(mapname,"r");
        fseek(fp,28,SEEK_SET);
        uint32 mx;
        fread((void*)(&mx),sizeof(uint32_t),1,fp);
        fseek(fp,40,SEEK_SET);
        float xlen;
        fread((void*)(&xlen),sizeof(float),1,fp);
        fclose(fp);
        if( xlen == 0 )
            return 1.0f;
        else
            return xlen/mx;
    }

    void set_apix (const char*mapname, const float apix, const uint32 X, const uint32 Y, const uint32 Z) {
        FILE*fp=fopen(mapname,"r+");
        uint32_t grid[3];
        grid[0] = X;
        grid[1] = Y;
        grid[2] = Z;
        fseek(fp,28,SEEK_SET);
        fwrite(grid,sizeof(uint32_t),3,fp);
        float apix_arr[3];
        apix_arr[0] = apix*grid[0];
        apix_arr[1] = apix*grid[1];
        apix_arr[2] = apix*grid[2];
        fseek(fp,40,SEEK_SET);
        fwrite(apix_arr,sizeof(float),3,fp);
        fclose(fp);
    }
	
    bool is_mode_float(const char*mapname) {
        FILE*fp=fopen(mapname,"r");
        fseek(fp,12,SEEK_SET);
        uint32 mode;
        fread((void*)(&mode),sizeof(uint32_t),1,fp);
        fclose(fp);
        if( mode == 2 )
            return true;
        else
            return false;
    }
	
    void read_size(uint32&X, uint32&Y, uint32&Z, const char*mapname) {
        FILE*fp=fopen(mapname,"r");
        fseek(fp,0,SEEK_SET);
        uint32 buf[3];
        fread((void*)buf,sizeof(uint32_t),3,fp);
        fclose(fp);
        X = buf[0];
        Y = buf[1];
        Z = buf[2];
    }
	
    void read(float*buffer, const uint32 X, const uint32 Y, const uint32 Z, const char*mapname) {
        if(!is_mode_float(mapname)) {
            fprintf(stderr,"Error: File %s is not mode FLOAT32.\n",mapname);
            exit(1);
        }
        FILE*fp=fopen(mapname,"rb");
        uint32 offset = 0;
        fseek(fp,92,SEEK_SET);
        fread((void*)(&offset),sizeof(uint32),1,fp);
        fseek(fp,1024+offset,SEEK_SET);
        size_t num_el = X*Y*Z;
        size_t read_el = fread((void*)buffer,sizeof(single),num_el,fp);
        fclose(fp);
        if( num_el != read_el ) {
            fprintf(stderr,"Error: File %s truncated.\n",mapname);
            exit(1);
        }
    }

    float *read(uint32&X, uint32&Y, uint32&Z, const char*mapname) {
        read_size(X,Y,Z,mapname);
        single *rslt = new single[X*Y*Z];
        read(rslt,X,Y,Z,mapname);
        return rslt;
    }
    
    void write(const single *data, const uint32 X, const uint32 Y, const uint32 Z, const char*mapname) {
        FILE*fp=fopen(mapname,"wb");
        uint32 header[256];
        for(int i=0;i<256;i++) header[i] = 0;
        header[0]  = X;
        header[1]  = Y;
        header[2]  = Z;
        header[3]  = 2;
        header[7]  = 1;
        header[8]  = 1;
        header[9]  = 1;
        header[16] = 1;
        header[17] = 2;
        header[18] = 3;
        fwrite((void*)header,sizeof(uint32),256,fp);
        fwrite((void*)data,sizeof(single),X*Y*Z,fp);
        fclose(fp);
    }

}

#endif 

