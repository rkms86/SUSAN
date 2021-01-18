function data = susan_read(filename)
% SUSAN_READ Reads a file according to its extension.
%   DATA = SUSAN_READ(FILENAME) reads FILENAME according to its extension.
%   Supported filetypes:
%   - .mrc, .st, .ali, .rec
%   - .tlt
%   - .xf
%   - .tbl
%   - .defocus
%   - .tomostxt

if( ~SUSAN.Utils.exist_file(filename) )
    error(['File ' filename ' does not exits or cannot be read.']);
end

% Read MRC formats:
if( SUSAN.Utils.is_extension(filename,'.mrc') || ...
    SUSAN.Utils.is_extension(filename,'.st') || ...
    SUSAN.Utils.is_extension(filename,'.ali') || ...
    SUSAN.Utils.is_extension(filename,'.rec') )
    data = SUSAN.IO.read_mrc(filename);
    
% Read TLT files:
elseif( SUSAN.Utils.is_extension(filename,'.tlt') )
    data = SUSAN.IO.read_tlt(filename);

% Read XF files:
elseif( SUSAN.Utils.is_extension(filename,'.xf') )
    data = SUSAN.IO.read_xf(filename);

% Read DEFOCUS files:
elseif( SUSAN.Utils.is_extension(filename,'.defocus') )
    data = SUSAN.IO.read_defocus(filename);

% Read TBL files:
elseif( SUSAN.Utils.is_extension(filename,'.tbl') )
    data = SUSAN.IO.read_dynamo_tbl(filename);

% Read TOMOSTXT files:
elseif( SUSAN.Utils.is_extension(filename,'.tomostxt') )
    data = SUSAN.Data.TomosInfo(filename);

% Unsupported files:
else
    error(['File ' filename ': unsupported extension.']);
end

end