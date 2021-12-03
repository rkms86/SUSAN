function enable_by_tilt(ptcls_in,tomos_list,ang)
% ENABLE_BY_TILT enables/disables projections by their tilt angle.
%   ENABLE_BY_TILT(PTCLS_IN,TOMOS_LIST,ANG) enables/disables the
%   projections on PTCLS_IN according the tilt angles from TOMO_LIST.
%   Enables the projections with tilts smaller than ANG, disables the rest.
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

    if( ~isa(tomos_list,'SUSAN.Data.TomosInfo') )
        error('Second argument must be a SUSAN.Data.TomosInfo object.');
    end

    if( ~(ang>0) )
        error('Third argument must be > 0.');
    end

    ptcls_in.prj_w = ParticlesGeom_enable_by_tilt_angle(ptcls_in.tomo_cix,tomos_list.proj_eZYZ,single(ang));
    ptcls_in.prj_w = ptcls_in.prj_w.*tomos_list.proj_weight(:,:,ptcls_in.tomo_cix+1);
	
end


