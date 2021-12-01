function [mrc_size,mrc_mode,mrc_apix] = get_mrc_info(filename)
% GET_MRC_INFO reads the info from header of an MRC file.
%   [MRC_SIZE, MRC_MODE, MRC_APIX] = GET_MRC_INFO(FILENAME) reads
%   the header of FILENAME, assuming is a MRC file, and returns
%   the size, mode and pixel size in MRC_SIZE, MRC_MODE and
%   MRC_APIX, respectively.

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

fp = fopen(filename,'rb');
mrc_size = fread(fp,[1,3],'uint32');

fseek(fp,12,'bof');
mrc_mode = fread(fp,[1,1],'uint32');

fseek(fp,28,'bof');
mx = fread(fp,[1,1],'uint32');
fseek(fp,40,'bof');
xlen = fread(fp,[1,1],'single');

if( xlen == 0 )
    mrc_apix = 1;
else
    mrc_apix = xlen/mx;
end

fclose(fp);

end
