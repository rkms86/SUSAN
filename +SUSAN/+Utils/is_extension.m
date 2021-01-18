function rslt = is_extension(file_name,file_ext)
% IS_EXTENSION checks if the a filename has a specific extension.
%   RSLT = IS_EXTENSION(FILENAME,FILEEXT) checks if FILENAME ends
%   with FILEEXT. Return true if it does, returns false otherwise.

    ext_len = length(file_ext);
    rslt = strcmp( file_name(max(1,end-ext_len+1):end), file_ext );

end
