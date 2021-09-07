classdef fsc_filter < handle

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

properties
    R single
    W single
end

methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function obj = fsc_filter(N)
    
        N = single(N);
        Nh = N/2;
        
        [X,Y,Z] = meshgrid(-Nh:(Nh-1),-Nh:(Nh-1),-Nh:(Nh-1));
        obj.R = min(sqrt( X.^2 + Y.^2 + Z.^2 ) + 1, Nh);
        obj.R = fftshift(obj.R);
        
        obj.W = ones(size(obj.R));
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_fsc_sqrt_FOM(obj,fsc_in)
    
        w = sqrt(max(2*fsc_in./(fsc_in+1),0));
        obj.W = interp1(w,obj.R);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_fsc_FOM(obj,fsc_in)
    
        w = 2*fsc_in./(fsc_in+1);
        obj.W = interp1(w,obj.R);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_fsc_optimal(obj,fsc_in,tau)
    
        if( nargin < 3 )
            tau = 0.1;
        end
        w = fsc_in./( fsc_in + tau*(1-fsc_in) );
        obj.W = interp1(w,obj.R);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_identity(obj)
    
        obj.W(:) = 1;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function add_bfactor(obj,bfactor)
    
        m = bfactor<0.001;
        w = bfactor./( bfactor.*bfactor );
        w(m) = 0;
        obj.W = obj.W.*interp1(w,obj.R);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function map_out = apply_filter(obj,map_in)
    
        map_out = ifftn( obj.W.*fftn(map_in) );
    end
end
    
end
