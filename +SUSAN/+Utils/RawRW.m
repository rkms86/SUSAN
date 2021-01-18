classdef RawRW

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
            fp = fopen(file_name,'rb');
        else
            error('File %s does not exist.',file_name);
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function fp = create_file(file_name,file_ext)
        
        %ext_len = length(file_ext);
        %if( ~strcmp( file_name(max(1,end-ext_len+1):end), file_ext ) )
        if( ~SUSAN.Utils.is_extension(file_name,file_ext) )
            file_name = [file_name '.' file_ext];
            warning('Filename set to %s',file_name);
        end
        
        fp = fopen(file_name,'wb');
        
    end
    
end
    
end
