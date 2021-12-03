function th = unimodal_th(pt_cloud,v1,v2)

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

v1 = repmat(v1,size(pt_cloud,1),1);
v2 = repmat(v2,size(pt_cloud,1),1);

a = v1 - v2;
b = pt_cloud - v2;

a(end,3) = 0;
b(end,3) = 0;

d = sqrt(sum(cross(a,b,2).^2,2)) ./ sqrt(sum(a.^2,2));

[~,ix_max] = max(d);

th = pt_cloud(ix_max,1);

end
