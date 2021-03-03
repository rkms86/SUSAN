#ifndef REF_MAPS_H
#define REF_MAPS_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "datatypes.h"
#include "reference.h"
#include "io.h"
#include "mrc.h"
#include "math_cpu.h"
#include "memory.h"

class RefMap {
public:
    single *map;
    single *half_A;
    single *half_B;
    single *mask;
    uint32 box_size;
    uint32 numel;

    bool   has_half_maps() { return (half_A != NULL) && (half_B != NULL); }
    bool   has_ref_map()   { return (map    != NULL); }
    bool   has_ref_mask()  { return (mask   != NULL); }

public:
    RefMap() {
        map    = NULL;
        half_A = NULL;
        half_B = NULL;
        mask   = NULL;

        box_size = 0;
        numel    = 0;
    }

    ~RefMap() {
		free_array(map   );
		free_array(half_A);
		free_array(half_B);
		free_array(mask  );
    }

    void load(const Reference&ref_info) {

        box_size = get_box_size(ref_info.map);

        if( box_size > 0 ) {

            numel = box_size*box_size*box_size;
            map   = load_mrc(ref_info.map);

            if( get_box_size(ref_info.h1) == box_size )
                half_A = load_mrc(ref_info.h1);

            if( get_box_size(ref_info.h2) == box_size )
                half_B = load_mrc(ref_info.h2);

            if( get_box_size(ref_info.mask) == box_size ) {
                mask = load_mrc(ref_info.mask);
                Math::mul(map,mask,numel);
                if(half_A!=NULL) Math::mul(half_A,mask,numel);
                if(half_B!=NULL) Math::mul(half_B,mask,numel);
            }

            float avg,std;
            if( mask != NULL ) {
                if( map != NULL ) {
                    Math::get_avg_std(avg,std,map,numel);
                    if( !Math::normalize(map,numel,avg,std) ) {
                        fprintf(stderr,"Error normalizing map %s\n",ref_info.map);
                        exit(1);
                    }
                    //Math::mul(map,mask,numel);
                }

                if( half_A != NULL ) {
                    Math::get_avg_std(avg,std,half_A,numel);
                    if( !Math::normalize(half_A,numel,avg,std) ) {
                        fprintf(stderr,"Error normalizing map %s\n",ref_info.h1);
                        exit(1);
                    }
                    //Math::mul(half_A,mask,numel);
                }

                if( half_B != NULL ) {
                    Math::get_avg_std(avg,std,half_B,numel);
                    if( !Math::normalize(half_B,numel,avg,std) ) {
                        fprintf(stderr,"Error normalizing map %s\n",ref_info.h2);
                        exit(1);
                    }
                    //Math::mul(half_A,mask,numel);
                }
            }

        }
    }

private:
    uint32 get_box_size(const char*fn) {

        uint32 X=0,Y=0,Z=0;

        if( check_mrc_exists(fn) ) {
            Mrc::read_size(X,Y,Z,fn);
            if( (X == Y) && (Y == Z) )
                return X;
            else
                return 0;
        }
        else
            return 0;
    }

    bool   check_mrc_exists(const char*fn) {
        bool rslt = false;
        if( fn[0] != 0 )
            if( IO::exists(fn) )
                rslt = true;
        return rslt;
    }

    single *load_mrc(const char*fn) {
        uint32 X=0,Y=0,Z=0;
        return Mrc::read(X,Y,Z,fn);
    }
};

#endif /// REF_MAPS_H


