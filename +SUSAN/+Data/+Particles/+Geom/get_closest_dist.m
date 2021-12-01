function min_dist = get_closest_dist(ptcls_in)     
% GET_CLOSEST_DIST returns a list with the minimum distance between
% particles.
%   MIN_DIST = GET_CLOSEST_DIST() returns an array with the minimum
%   distance of each particle to the other ones.
%
%   See also SUSAN.Data.ParticlesInfo.

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

    if( ~isa(ptcls_in,'SUSAN.Data.ParticlesInfo') )
        error('First argument must be a SUSAN.Data.ParticlesInfo object.');
    end
    
    min_dist = [];

    tomos_id = unique( ptcls_in.tomo_id );

    for i = 1:length(tomos_id)
        t_idx = (ptcls_in.tomo_id == tomos_id(i));
        data = [ptcls_in.position(t_idx,:)+ptcls_in.ali_t(t_idx,:)];
        cur_dist = ParticlesGeom_calc_min_dist(data');
        min_dist = [min_dist; cur_dist];
    end
end
