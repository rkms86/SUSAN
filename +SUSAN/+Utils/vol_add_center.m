function vol_out = vol_add_center(big_vol, small_vol)

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

vol_in = conditional_load_mrc(small_vol);

ix_x = (1:size(small_vol,1)) + ceil( (size(big_vol,1)-size(small_vol,1))/2 );
ix_y = (1:size(small_vol,2)) + ceil( (size(big_vol,2)-size(small_vol,2))/2 );
ix_z = (1:size(small_vol,3)) + ceil( (size(big_vol,3)-size(small_vol,3))/2 );

vol_out = big_vol;

vol_out(ix_x,ix_y,ix_z) = small_vol;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function v = conditional_load_mrc(arg)

if( ischar(arg) )
    v = SUSAN.IO.read_mrc(arg);
else
    v = arg;
end

end
