function new_file_name = force_extension(file_name,file_ext)
% FORCE_EXTENSION checks if a filename has an extension and appends it if not.
%   NEWFILENAME = FORCE_EXTENSION(FILENAME,FILEEXT) checks if FILENAME ends
%   with FILEEXT, and appends it if not.

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
