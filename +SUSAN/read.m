function data = read(filename)
% READ Reads a file according to its extension.
%   DATA = READ(FILENAME) reads FILENAME according to its extension.
%   Supported filetypes:
%   - .mrc, .st, .ali, .rec
%   - .tlt
%   - .xf
%   - .tbl
%   - .defocus
%   - .tomostxt

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file is part of the Substack Analysis (SUSAN) framework.
% Copyright (c) 2018-2021 Ricardo Miguel Sanchez Loayza.
% Max Planck Institute of Biophysics
% Department of Structural Biology - Kudryashev Group.
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU Affero General Public License as
% published by the Free Software Foundation, either version 3 of the
% License, or (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU Affero General Public License for more details.
% 
% You should have received a copy of the GNU Affero General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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

% Read PTCLSRAW files:
elseif( SUSAN.Utils.is_extension(filename,'.ptclsraw') )
    data = SUSAN.Data.ParticlesInfo(filename);

% Unsupported files:
else
    error(['File ' filename ': unsupported extension.']);
end

end
