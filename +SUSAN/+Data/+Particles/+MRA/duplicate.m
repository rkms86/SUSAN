function duplicate(ptcls_in,ref_ix)
% DUPLICATE Copies a reference alignment info and place it at the end.
%   DUPLICATE(PTCLS,REF_IX) Copies the alignment information for the class/
%   reference REF_IX and appends it to the alignment matrices.

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
    
    if( nargin < 2 )
        ref_ix = 1;
    end

    ptcls_in.ali_eZYZ = cat(3, ptcls_in.ali_eZYZ, ptcls_in.ali_eZYZ(:,:,ref_ix));
    ptcls_in.ali_t    = cat(3, ptcls_in.ali_t   , ptcls_in.ali_t   (:,:,ref_ix));
    ptcls_in.ali_cc   = cat(3, ptcls_in.ali_cc  , ptcls_in.ali_cc  (:,:,ref_ix));
    ptcls_in.ali_w    = cat(3, ptcls_in.ali_w   , ptcls_in.ali_w   (:,:,ref_ix));

end
