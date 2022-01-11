function write_mrc(map,filename,apix)
% Writes a MRC file.
%   WRITE_MRC(MAP,FILENAME) Saves MAP in FILENAME using the MRC format.
%   WRITE_MRC(...,APIX) Additionally, it writes the pixel size in the file.

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

if( nargin < 3 )
    apix = 1;
end

fp = fopen(filename,'w+');

% Empty headers
fwrite(fp,zeros(1,1024,'uint8'));

% Save content
fwrite(fp,single(map),'float32');

% Save size and Mode:
fseek(fp, 0, 'bof');
buffer = zeros(1,4,'uint32');
if( ndims(map) == 3 )
    buffer(:) = [uint32(size(map)), 2];
else
    buffer(:) = [uint32(size(map)), 1, 2];
end
fwrite(fp,buffer,'uint32');

% Save Pixel Size:
xyz_len = single(size(map));
xyz_len = apix*xyz_len;
fseek(fp, 28, 'bof');
buffer = zeros(1,3,'uint32');
if( ndims(map) == 3 )
    buffer(:) = uint32(size(map));
else
    buffer(:) = [uint32(size(map)), 1];
end
fwrite(fp,buffer,'uint32');
fseek(fp, 40, 'bof');
if( ndims(map) == 3 )
    fwrite(fp,xyz_len,'float32');
else
    fwrite(fp,[xyz_len 1],'float32');
end
fwrite(fp,[90 90 90],'float32');

% Save Mapping
fseek(fp, 64, 'bof');
buffer = zeros(1,3,'uint32');
buffer(:) = [1 2 3];
fwrite(fp,buffer,'uint32');

% Save MRC map id
fseek(fp, 208, 'bof');
buffer = zeros(1,8,'uint8');
buffer(:) = ['MAP DD' 0 0];
fwrite(fp,buffer,'uint8');

% finished
fclose(fp);


end


