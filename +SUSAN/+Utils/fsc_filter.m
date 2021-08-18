classdef fsc_filter < handle

properties
    R single
    W single
end

methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function obj = fsc_filter(N)
    
        N = single(N);
        Nh = N/2;
        obj.W = ones(Nh,1);
        
        [X,Y,Z] = meshgrid(-Nh:(Nh-1),-Nh:(Nh-1),-Nh:(Nh-1));
        obj.R = min(sqrt( X.^2 + Y.^2 + Z.^2 ) + 1, Nh);
        obj.R = fftshift(obj.R);
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