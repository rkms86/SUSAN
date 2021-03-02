classdef CtfEstimator < handle
% SUSAN.Modules.CtfEstimator estimates the defocus values.
%    Estimates the defocus values .
%
% SUSAN.Modules.CtfEstimator Properties:
%    gpu_list   - (vector) list of GPU IDs to use for alignment.
%    binning    - (scalar) frequency range to use.
%    resolution - (struct) resolution's search range (angstroms min/max).
%    defocus    - (struct) defocus' search range (angstroms min/max).
%    tlt_range  - (scalar) defocus' search range between tilts (angstroms).
%    refine_def - (struct) defocus' refinement (angstroms range/step).
%
% SUSAN.Modules.CtfEstimator Methods:
%    CtfEstimator - (Constructor) creates a CtfEstimator with a given box size.
%    estimate     - perfoms the CTF estimation.
%    show_cmd     - returns the command to execute the CTF estimation.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
properties(SetAccess=private)
    box_size = 512;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
properties
    gpu_list    uint32 = 0;
    binning     uint32 = 0;
    resolution         = struct('min',40,'max',7);
    defocus            = struct('min',10000,'max',90000);
    tlt_range   single = 3000;
    refine_def         = struct('range',2000,'step',100);
    bfactor_max single = 600;
    res_thres   single = 0.75;
    verbose     uint32 = 0;
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

methods
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function obj = CtfEstimator(in_size)
    % CtfEstimator [Constructor] Creates CtfEstimator object:
    %   CtfEstimator(IN_SIZE) Creates a CtfEstimator object that will use
    %   a patch size of IN_SIZE.
    %
    %   See also SUSAN.Data.TomosInfo.
    
        obj.box_size = in_size;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function tomos = estimate(obj,output_dir,ptcls_file,tomos_file)
    % ESTIMATE perfoms the CTF estimation.
    %   TOMOS = ESTIMATE(OUT_DIR,PTCLS_IN,TOMOS_IN) estimates the defocus 
    %   values using the information on PTCLS_IN and TOMOS_IN and saves 
    %   the output files on the OUT_DIR folder. It returns an updated
    %   TomosInfo object with the new defocus values.
    %
    %   See also SUSAN.Data.ParticlesInfo and SUSAN.Data.TomosInfo.
        
        if( output_dir(end) == '/' )
            output_dir = output_dir(1:end-1);
        end
    
        temp_tomofile = false;
        if( ~ischar(tomos_file) )
            if( isa(tomos_file,'SUSAN.Data.TomosInfo') )
                tomos_file.save('.tmp.tomostxt');
                tomos_file = '.tmp.tomostxt';
                temp_tomofile = true;
            else
                error('TOMOS_FILE must be a char array or a TOMOSINFO object.');
            end
        end
        
        ctf_exec = obj.show_cmd(output_dir,ptcls_file,tomos_file);
        
        stat = system(ctf_exec);
        
        if( stat == 0 )
            tomos = SUSAN.Data.TomosInfo(tomos_file);
            for i = 1:length(tomos.tomo_id)
                new_defocus_file = sprintf([output_dir '/Tomo%03d/defocus.txt'],tomos.tomo_id(i));
                if( SUSAN.Utils.exist_file(new_defocus_file) ) 
                    fprintf('Updating tomogram with ID %d.\n',tomos.tomo_id(i));
                    tomos.set_defocus(i,new_defocus_file)
                end
            end
        else
            error('CtfEstimator crashed.');
        end
        
        if( temp_tomofile )
            system(['rm ' tomos_file]);
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function cmd = show_cmd(obj,output_dir,ptcls_file,tomos_file)
    % SHOW_CMD returns the command to execute the CTF estimation.
    %   CMD = SHOW_CMD(OUT_DIR,PTCLS_IN,TOMOS_IN) returns the command to 
    %   execute the CTF estimation.
    %
    %   See also SUSAN.Data.ParticlesInfo and SUSAN.Data.TomosInfo.
        
        if( output_dir(end) == '/' )
            output_dir = output_dir(1:end-1);
        end
        
        if( ~ischar(ptcls_file) )
            error('PTCLS_FILE must be a char array.');
        end
        
        if( ~ischar(tomos_file) )
            if( isa(tomos_file,'SUSAN.Data.TomosInfo') )
                tomos_file = '.tmp.tomostxt';
            else
                error('TOMOS_FILE must be a char array or a TOMOSINFO object.');
            end
        end
        
        gpu_list_str = sprintf('%d,',obj.gpu_list);
        gpu_list_str = gpu_list_str(1:end-1);
        
        num_threads = length(obj.gpu_list);
        
        cmd = [SUSAN.bin_path '/susan_estimate_ctf'];        
        cmd = [cmd ' -tomos_in '    tomos_file];
        cmd = [cmd ' -data_out '    output_dir];
        cmd = [cmd ' -ptcls_file '  ptcls_file];
        cmd = [cmd ' -box_size '    sprintf('%d',obj.box_size)];
        cmd = [cmd ' -n_threads '   sprintf('%d',num_threads)];
        cmd = [cmd ' -res_range '   sprintf('%f,%f',obj.resolution.min,obj.resolution.max)];
        cmd = [cmd ' -res_thres '   sprintf('%f',obj.res_thres)];
        cmd = [cmd ' -def_range '   sprintf('%f,%f',obj.defocus.min,obj.defocus.max)];
        cmd = [cmd ' -tilt_search ' sprintf('%f',obj.tlt_range)];
        cmd = [cmd ' -refine_def '  sprintf('%f,%f',obj.refine_def.range,obj.refine_def.step)];
        cmd = [cmd ' -binning '     sprintf('%d',obj.binning)];
        cmd = [cmd ' -gpu_list '    gpu_list_str];
        cmd = [cmd ' -bfactor_max ' sprintf('%f',obj.bfactor_max)];
        cmd = [cmd ' -verbose '     sprintf('%d',obj.verbose)];
        
    end
    
    
end


end

