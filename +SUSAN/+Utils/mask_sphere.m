function msk = mask_sphere(rad,siz,center)
% MASK_SPHERE creates a spherical mask.
%   MSK = MASK_SPHERE(RAD,SIZ) Creates a mask of radius RAD in
%   a volume of size SIZ.
%   MSK = MASK_SPHERE(...,CENTER) Sets the center of the sphere
%   in CENTER.

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

