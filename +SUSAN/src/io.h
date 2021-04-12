#ifndef IO_H
#define IO_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <sys/stat.h>
#include <ftw.h>
#include <unistd.h>

#include <errno.h>

#include "datatypes.h"

namespace IO {

////

bool exists(const char*filename) {
    struct stat buffer;
    return ( stat( filename, &buffer) == 0 );
}

bool exist_dir(const char*dirname) {
    struct stat buffer;
    return ( stat( dirname, &buffer) == 0 ) && ( S_ISDIR(buffer.st_mode) );
}

int unlink_cb(const char *fpath, const struct stat *sb, int typeflag, struct FTW *ftwbuf) {
    int rv = remove(fpath);
    if( rv ) perror(fpath);
    return rv;
}

void create_dir(const char*filename) {
    if( !exist_dir(filename) )
        mkdir(filename, 0755);
}

void delete_dir(const char*filename) {
    if( exists(filename) )
        nftw(filename, unlink_cb, 64, FTW_DEPTH | FTW_PHYS);
}

bool check_file_extension(const char*filename,const char*extension) {

    int fn_len  = strlen(filename);
    int ext_len = strlen(extension);

    return (strcmp( filename-ext_len+fn_len, extension )==0);

}

////

bool check_fread(void *ptr, size_t size, size_t nmemb, FILE *fp) {
    size_t read_nmemb = fread(ptr,size,nmemb,fp);
    if( read_nmemb != nmemb ) {
        fprintf(stderr,"Error reading file: %ld/%ld (%ld) [%d/%d]\n",read_nmemb,nmemb,size,ferror(fp),errno);
        return false;
    }
    return true;
}

////

namespace DefocusIO {
	
	bool read(Defocus&def,FILE*fp) {
		int n = 0;
		n += fscanf(fp,"%f",&def.U);
		n += fscanf(fp,"%f",&def.V);
		n += fscanf(fp,"%f",&def.angle);
		n += fscanf(fp,"%f",&def.ph_shft);
		n += fscanf(fp,"%f",&def.Bfactor);
		n += fscanf(fp,"%f",&def.ExpFilt);
		n += fscanf(fp,"%f",&def.max_res);
		n += fscanf(fp,"%f",&def.score);
		return (n==8);
	}
	
	void write(FILE*fp,const Defocus&def) {
		fprintf(fp,"%10.2f  %10.2f  %8.3f  %8.3f  %8.2f  %8.2f  %6.3f  %e\n",def.U,def.V,def.angle,def.ph_shft,def.Bfactor,def.ExpFilt,def.max_res,def.score);
	}
	
}

////

class TxtParser {

protected:
    char buffer[ SUSAN_FILENAME_LENGTH ];
    FILE *fp;

public:
    TxtParser(const char*filename,const char*extension) {
        if( IO::exists( filename ) ) {
            if( IO::check_file_extension(filename,extension) ) {
                fp = fopen(filename,"r");
            }
            else {
                fprintf(stderr,"File %s: wrong extension (%s).\n",filename,extension);
                exit(0);
            }
        }
        else {
            fprintf(stderr,"File %s does not exist.\n",filename);
            exit(0);
        }
    }

    ~TxtParser() {
        fclose(fp);
    }

    void get_value(uint32&value,const char*tag) {
        int tmp;
        get_value(tmp,tag);
        value = (uint32)tmp;
    }

    void get_value(int&value,const char*tag) {
        value = atoi( get_char_ptr(tag) );
    }

    void get_value(float&value,const char*tag) {
        value = atof( get_char_ptr(tag) );
    }

    void get_value(VUInt3&value,const char*tag) {
        char*ptr = get_char_ptr(tag);
        remove_commas(ptr);
        if( sscanf(ptr,"%d %d %d",&value.x,&value.y,&value.z) != 3 ) {
            fprintf(stderr,"Error parsing %s: %s\n",tag,ptr);
        }
    }

    void get_value(Vec3&value,const char*tag) {
        char*ptr = get_char_ptr(tag);

        if( sscanf(ptr,"%f %f %f",&value.x,&value.y,&value.z) != 3 ) {
            fprintf(stderr,"Error parsing %s: %s\n",tag,ptr);
        }
    }

    void get_str(char*str,const char*tag) {
        char*ptr = get_char_ptr(tag);
        if( ptr[0] > 0 )
            strcpy(str,ptr);
        else
            str[0] = 0;
    }

    char*read_line_raw() {
        get_line();
        return buffer;
    }

protected:
    void get_line() {
        bool read_next_line = true;
        while( read_next_line ) {
            if( fgets(buffer,SUSAN_FILENAME_LENGTH,fp) != NULL ) {
                int buf_len = strlen(buffer);
                if( buf_len > 0 ) {
                    if( buffer[0] != '#' ) {
                        read_next_line = false;
                        if( buffer[buf_len-1] == '\n' )
                            buffer[buf_len-1] = 0;
                    }
                }
            }
            else {
                fprintf(stderr,"Truncated file\n");
                exit(0);
            }
        }
    }

    char*get_char_ptr(const char*tag) {
        char*rslt = buffer;
        int tag_len = strlen(tag);
        get_line();
        if( strncmp( buffer, tag, tag_len ) == 0 ) {
            rslt = buffer+tag_len+1;
        }
        else {
            fprintf(stderr,"Requested tag %s, buffer: %s\n",tag,buffer);
            exit(0);
        }
        return rslt;
    }

    void remove_commas(char*buf) {
        for(int i=0;i<SUSAN_FILENAME_LENGTH;i++) {
            if( buf[i] == 0 )
                break;
            if( buf[i] == ',' )
                buf[i] = ' ';
        }
    }

};

////

uint32 parse_uint32_strlist(uint32* &ptr, const char*strlist) {

    int i = 0;
    uint32 count = 1;

    while( strlist[i] > 0 ) {
        if( strlist[i] == ',' )
            count++;
        i++;
    }

    ptr = new uint32[count];

    char buffer[50];
    int j = 0, k = 0;;
    i = 0;
    while( strlist[i] > 0 ) {
        if( strlist[i] == ',' ) {
            buffer[k] = 0;
            ptr[j] = atoi(buffer);
            k = 0;
            j++;
        }
        else {
            buffer[k] = strlist[i];
            k++;
        }
        i++;
    }
    buffer[k] = 0;
    ptr[j] = atoi(buffer);

    return count;
}

uint32 parse_single_strlist(single* &ptr, const char*strlist) {

    int i = 0;
    uint32 count = 1;

    while( strlist[i] > 0 ) {
        if( strlist[i] == ',' )
            count++;
        i++;
    }

    ptr = new single[count];

    char buffer[50];
    int j = 0, k = 0;;
    i = 0;
    while( strlist[i] > 0 ) {
        if( strlist[i] == ',' ) {
            buffer[k] = 0;
            ptr[j] = atof(buffer);
            k = 0;
            j++;
        }
        else {
            buffer[k] = strlist[i];
            k++;
        }
        i++;
    }
    buffer[k] = 0;
    ptr[j] = atof(buffer);

    return count;
}

class ProgressReporter {

protected:
    char   buf_clr[60];
    char   buf_inf[60];
    bool   should_clean;
    uint32 prev_particles;
    uint32 total_particles;

    single part_per_second_arr[4];
    int    arr_occupancy;
    int    arr_ix;

    bool   is_first;

public:
    ProgressReporter(const char*title,uint32 tot_ptcls) {
        prev_particles  = 0;
        total_particles = tot_ptcls;
        arr_occupancy   = 0;
        arr_ix          = 0;

        for(int i=0;i<59;i++)
            buf_clr[i] = '\b';
        buf_clr[59] = 0;

        printf("    %s: ",title);
        fflush(stdout);

        is_first = true;
        should_clean = false;
    }

    void show_progress(const uint32 particles, const bool are_processing) {

        clear_output();

        single progress = 100.0*( (single)particles )/( (single)total_particles );

        if( particles >= total_particles )
            set_output_done();
        else if( !are_processing )
            set_output_buffering(progress);
        else {
            single ptcls_per_sec = get_ptcls_per_sec(particles);
            single eta_sec = get_eta_sec(particles,ptcls_per_sec);
            set_output_progress(progress,ptcls_per_sec,eta_sec);
        }

        fprintf(stdout,buf_inf);
        should_clean = true;
        fflush(stdout);
    }

protected:
    void clear_output() {
        if( should_clean ) {
            fprintf(stdout,buf_clr);
            fflush(stdout);
        }
    }

    void set_output_done() {
        sprintf(buf_inf,"100.00%% [Done]");
        pad_output_buf();
    }

    void set_output_buffering(const single progress) {
        sprintf(buf_inf,"%6.2f%% [Buffering]",progress);
        pad_output_buf();
    }

    void set_output_progress(const single progress, const single ptcls_per_sec, const single eta_sec) {
        char eta_txt[20];
        get_eta_txt(eta_txt,eta_sec);
        sprintf(buf_inf,"%6.2f%% [%.1f particles/sec | ETA: %s]",progress,ptcls_per_sec,eta_txt);
        pad_output_buf();
    }

    void pad_output_buf() {
        for(int i=strlen(buf_inf);i<59;i++)
            buf_inf[i] = ' ';
        buf_inf[59] = 0;
    }

    single get_ptcls_per_sec(const uint32 particles) {

        uint32 part_per_sec = particles - prev_particles;
        prev_particles = particles;

        part_per_second_arr[arr_ix] = (single)part_per_sec;

        if( is_first ) {
            is_first = false;
            return part_per_second_arr[arr_ix];
        }

        arr_ix++;
        if(arr_ix==4) arr_ix = 0;

        arr_occupancy++;
        if(arr_occupancy>4) arr_occupancy = 4;

        single rslt = 0;
        for(int i=0;i<arr_occupancy;i++) {
            rslt += part_per_second_arr[i];
        }

        return rslt/((single)arr_occupancy);
    }

    single get_eta_sec(const uint32 particles, const single ptcls_per_sec) {
        single remaining_ptcls = total_particles - particles;
        return remaining_ptcls/ptcls_per_sec;
    }

    void get_eta_txt(char*eta_txt, const single eta_sec) {
        single eta_w = eta_sec;
        single n_days = floorf( eta_w/86400 );
        eta_w = eta_w - n_days*86400;
        single n_hour = floorf( eta_w/3600 );
        eta_w = eta_w - n_hour*3600;
        single n_mins = floorf( eta_w/60 );
        single n_secs = eta_w - n_mins*60;

        if( n_days > 0 ) {
            sprintf(eta_txt,"%.0fd %02.0f:%02.0f:%02.0f",n_days,n_hour,n_mins,n_secs);
        }
        else {
            sprintf(eta_txt,"%02.0f:%02.0f:%02.0f",n_hour,n_mins,n_secs);
        }
    }
};

}

#endif


