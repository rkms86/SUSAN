function def = read_defocus(defname)
% READ_DEFOCUS Reads a IMOD's defocus file.
%   DEF = READ_DEFOCUS(DEFNAME) Reads DEFNAME as a matrix with the defocus
%   values (U,V,angle).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file is part of the Substack Analysis (SUSAN) framework.
% Copyright (c) 2018-2021 Ricardo Miguel Sanchez Loayza.
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

fp = fopen(defname,'r');

cur_line = fgetl(fp);

if ~ischar(cur_line)
    error(['Invalid file: ' defname]);
end

first_line = sscanf(cur_line,'%f');
def_version = first_line(end);

if( def_version == 3 )
    def = defocus_v3(fp);
elseif( def_version == 2 )
    fseek(fp,0,'bof');
    def = defocus_v2(fp);
else
    error('Unknown/invalid defocus version: %d\n',def_version);
end

fclose(fp);

end

%%

function def = defocus_v2(fp)

def = [];
cur_line = fgetl(fp);
while ischar(cur_line)
    split_line = sscanf(cur_line,'%f');
    defU = 10*split_line(5);
    def(end+1,:) = [defU defU 0];
    cur_line = fgetl(fp);
end

end

%%

function def = defocus_v3(fp)

def = [];
cur_line = fgetl(fp);
while ischar(cur_line)
    split_line = sscanf(cur_line,'%f');
    defU = 10*split_line(5);
    defV = 10*split_line(6);
    defA = split_line(7);
    def(end+1,:) = [defU defV defA];
    cur_line = fgetl(fp);
end

end
