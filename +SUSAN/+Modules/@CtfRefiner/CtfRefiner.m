classdef CtfRefiner < handle
% SUSAN.Modules.CtfRefiner refines the defocus values.
%    Refines the defocus value per particle per projection.
%
% SUSAN.Modules.CtfRefiner Properties:
%    gpu_list     - (vector) list of GPU IDs to use for alignment.
%    resolution   - (struct) resolution's search range (angstroms min/max).
%    defocus      - (struct) defocus' search range (angstroms min/max).
%    astgm_search - (struct) defocus' astigmatism values (angle/angstroms).
%    verbose_id   - (scalar) create debug files for the particle ID
%
% SUSAN.Modules.CtfRefiner Methods:
%    CtfRefiner - (Constructor) creates a CtfRefiner with a given box size.
%    refine     - perfoms the CTF refinement.
%    show_cmd   - returns the command to execute the CTF refinement.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
properties(SetAccess=private)
    box_size = 512;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
properties
    gpu_list    uint32 = 0;
    resolution         = struct('min',40,'max',7);
    defocus            = struct('min',10000,'max',90000);
    astgm_search       = struct('angle',10,'def',1000);
    res_thres   single = 0.75;
    verbose_id  int32  = -1;
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

methods
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function obj = CtfRefiner(in_size)
    % CtfRefiner [Constructor] Creates CtfRefiner object:
    %   CtfRefiner(IN_SIZE) Creates a CtfRefiner object that will use
    %   a patch size of IN_SIZE.
    %
    %   See also SUSAN.Data.TomosInfo.
    
        obj.box_size = in_size;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function refine(obj,ptcls_out,ptcls_in,tomos_file)
    % REFINE perfoms the CTF refinement.
    %   REFINE(PTCLS_OUT,PTCLS_IN,TOMOS_IN) refine the defocus values from 
    %   PTCLS_IN and TOMOS_IN and saves them in PTCLS_OUT.
    %
    %   See also SUSAN.Data.ParticlesInfo and SUSAN.Data.TomosInfo.
        
        ctf_exec = obj.show_cmd(ptcls_out,ptcls_in,tomos_file);
        
        stat = system(ctf_exec);
        
        if( stat ~= 0 )
            error('CtfEstimator crashed.');
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function cmd = show_cmd(obj,ptcls_out,ptcls_in,tomos_file)
    % SHOW_CMD returns the command to execute the CTF refinement.
    %   CMD = SHOW_CMD(PTCLS_OUT,PTCLS_IN,TOMOS_IN) returns the command to 
    %   execute the CTF refinement.
    %
    %   See also SUSAN.Data.ParticlesInfo and SUSAN.Data.TomosInfo.
        
        if( ~ischar(ptcls_out) )
            error('PTCLS_OUT must be a char array.');
        end
        
        if( ~ischar(ptcls_out) )
            error('PTCLS_IN must be a char array.');
        end
        
        if( ~ischar(tomos_file) )
            error('TOMOS_FILE must be a char array.');
        end
        
        gpu_list_str = sprintf('%d,',obj.gpu_list);
        gpu_list_str = gpu_list_str(1:end-1);
        
        num_threads = length(obj.gpu_list);
        
        if( isdeployed )
            cmd = 'susan_refine_ctf';
        else
            cmd = [SUSAN.bin_path '/susan_refine_ctf'];
        end
        cmd = [cmd ' -tomos_file '  tomos_file];
        cmd = [cmd ' -ptcls_in '    ptcls_in];
        cmd = [cmd ' -ptcls_out '   ptcls_out];
        cmd = [cmd ' -box_size '    sprintf('%d',obj.box_size)];
        cmd = [cmd ' -n_threads '   sprintf('%d',num_threads)];
        cmd = [cmd ' -res_range '   sprintf('%f,%f',obj.resolution.min,obj.resolution.max)];
        cmd = [cmd ' -def_range '   sprintf('%f,%f',obj.defocus.min,obj.defocus.max)];
        cmd = [cmd ' -astg_range '  sprintf('%f,%f',obj.astgm_search.angle,obj.astgm_search.def)];
        cmd = [cmd ' -gpu_list '    gpu_list_str];
        if( obj.verbose_id >= 0 )
            cmd = [cmd ' -verbose_id '     sprintf('%d',obj.verbose_id)];
        end
        
    end
    
    
end


end

