function new_file_name = force_extension(file_name,file_ext)
% FORCE_EXTENSION checks if a filename has an extension and appends it if not.
%   NEWFILENAME = FORCE_EXTENSION(FILENAME,FILEEXT) checks if FILENAME ends
%   with FILEEXT, and appends it if not.

    new_file_name = file_name;

    if( ~SUSAN.Utils.is_extension(new_file_name,file_ext) )
        if( file_ext(1) == '.' )
            new_file_name = [file_name file_ext];
        else
            new_file_name = [file_name '.' file_ext];
        end
        warning(['Changing ' file_name ' to ' new_file_name]);
    end

end
