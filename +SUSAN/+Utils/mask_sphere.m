function msk = mask_sphere(rad,siz,center)
% MASK_SPHERE creates a spherical mask.
%   MSK = MASK_SPHERE(RAD,SIZ) Creates a mask of radius RAD in
%   a volume of size SIZ.
%   MSK = MASK_SPHERE(...,CENTER) Sets the center of the sphere
%   in CENTER.

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
    center = siz/2+1;
end

if( isempty(center) )
    center = siz/2+1;
end
   
if( length(center) ~= 3 )
    center = [ center(1) center(1) center(1) ];
end

msk = mask_sphere_core(single(rad),single(siz),single(center-1));

end

