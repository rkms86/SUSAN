function write_dynamo_tbl(tbl,tablename)
% WRITE_DYNAMO_TBL Writes a DYNAMO table.
%   WRITE_DYNAMO_TBL(TBL,TABLENAME) Saves TBL as a DYNAMO table with the
%   name TABLENAME.
%
%   See also dthelp.

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

dlmwrite(tablename,real(tbl),' ');

end
