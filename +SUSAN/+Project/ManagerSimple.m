classdef ManagerSimple < SUSAN.Project.Manager

methods

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_angular_search(obj,c_r,c_s,i_r,i_s,ref_lvl,ref_factor)
        
        if( nargin < 7 )
            ref_factor = 1;
        end
        
        if( nargin < 6 )
            ref_lvl = 0;
        end
        
        obj.aligner.set_angular_search(c_r,c_s,i_r,i_s);
        obj.aligner.set_angular_refinement(ref_lvl,ref_factor);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_offset_ellipsoid(obj,off_range,off_sampling,should_drift)
        
        if( nargin < 4 )
            should_drift = false;
        end
        
        if( nargin < 3 )
            off_sampling = 1;
        end
        
        obj.aligner.drift = should_drift;
        obj.aligner.set_offset_ellipsoid(off_range,off_sampling);
        
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_offset_cylinder(obj,off_range,off_sampling,should_drift)
        
        if( nargin < 4 )
            should_drift = false;
        end
        
        if( nargin < 3 )
            off_sampling = 1;
        end
        
        obj.aligner.drift = should_drift;
        obj.aligner.set_offset_cylinder(off_range,off_sampling);
        
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_rec_ssnr(obj,ssnr_s,ssnr_f)
        
        if( nargin < 3 )
            ssnr_f = 0;
        end
        
        obj.averager.ssnr_s = ssnr_s;
        obj.averager.ssnr_f = ssnr_f;        
    end
    
end
    
end
