classdef RawRW

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
