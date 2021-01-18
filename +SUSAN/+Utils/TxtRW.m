classdef TxtRW

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

methods(Static)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function fp = open_existing_file(file_name,file_ext)
        
        %ext_len = length(file_ext);
        %if( ~strcmp( file_name(max(1,end-ext_len+1):end), file_ext ) )
        if( ~SUSAN.Utils.is_extension(file_name,file_ext) )
            error('Wrong file extension (it should be %s).',file_ext);
        end
        
        if( exist(file_name,'file') )
            fp = fopen(file_name,'r');
        else
            error('File %s does not exist.',file_name);
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function fp = create_file(file_name,file_ext)
        
        %ext_len = length(file_ext);
        %if( ~strcmp( file_name(max(1,end-ext_len+1):end), file_ext ) )
        file_name_work = file_name;
        if( ~SUSAN.Utils.is_extension(file_name,file_ext) )
            file_name_work = [file_name '.' file_ext];
                warning('Filename set to %s',file_name_work);
        end
        
        fp = fopen(file_name_work,'w');
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function value_exists = write_pair_uint(fp,tag,val)
        
        if( isempty(val) )
            value_exists = false;
        else
            fprintf(fp,'%s:%d\n',tag,uint32(val));
            value_exists = true;
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function value_exists = write_pair_single(fp,tag,val)
        
        if( isempty(val) )
            value_exists = false;
        else
            fprintf(fp,'%s:%f\n',tag,single(val));
            value_exists = true;
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function value_exists = write_pair_uint_arr(fp,tag,arr)
        
        if( isempty(arr) )
            value_exists = false;
        else
            tmp = sprintf('%d,',arr);
            fprintf(fp,'%s:%s\n',tag,tmp(1:end-1));
            value_exists = true;
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function value_exists = write_pair_char(fp,tag,val)
        
        if( isempty(val) )
            value_exists = false;
        else
            fprintf(fp,'%s:%s\n',tag,val);
            value_exists = true;
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function write_optional_pair_char(fp,tag,val)
        
        if( isempty(val) )
            fprintf(fp,'%s:\n',tag);
        else
            fprintf(fp,'%s:%s\n',tag,val);
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function cur_line = read_line(fp)
        
        read_next_line = true;
        while( read_next_line )
            cur_line = fgetl(fp);
            if( cur_line < 0 )
                error('Truncated file');
            elseif( ~isempty(cur_line) )
                if( cur_line(1) ~= '#' )
                    read_next_line = false;
                end
            end
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [tag,val] = read_pair(fp)
        
        tag = '';
        val = '';
        tmp = SUSAN.Utils.TxtRW.read_line(fp);
        ix_sep = strfind(tmp,':');
        if( ~isempty(ix_sep) )
            tag = strtrim(tmp(  1:ix_sep(1)-1));
            val = strtrim(tmp(ix_sep(1)+1:end));
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [val] = read_tag_char(fp,tag)
        [tag_read,val] = SUSAN.Utils.TxtRW.read_pair(fp);
        if( strcmp(tag_read,tag) )
        else
            error('Requested tag %s, found %s.',tag,tag_read);
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [val] = read_tag_double(fp,tag)
        val = str2double( SUSAN.Utils.TxtRW.read_tag_char(fp,tag) );
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [val] = read_tag_int(fp,tag)
        val = int32( SUSAN.Utils.TxtRW.read_tag_double(fp,tag) );
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [val] = read_tag_double_arr(fp,tag)
        val = str2double( strsplit( SUSAN.Utils.TxtRW.read_tag_char(fp,tag), ',' ) );
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [val] = read_tag_int_arr(fp,tag)
        val = int32( SUSAN.Utils.TxtRW.read_tag_double_arr(fp,tag) );
    end
    
end
    
end
