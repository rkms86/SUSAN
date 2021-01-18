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
%    work_dir   - (string) working directory for temporarily files.
%
% SUSAN.Modules.CtfEstimator Methods:
%    CtfEstimator - (Constructor) creates a CtfEstimator with a given box size.
%    estimate     - perfoms the CTF estimation.

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
    work_dir           = 'tmp/';
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
        
        if( ~exist(obj.work_dir,'dir') )
            mkdir(obj.work_dir);
        end
        
        if( obj.work_dir(end) ~= '/' )
            obj.work_dir = [obj.work_dir '/'];
        end
        
        if( ~exist(output_dir,'dir') )
            mkdir(output_dir);
        end
        
        if( output_dir(end) ~= '/' )
            output_dir = [output_dir '/'];
        end
        
        if( ~ischar(ptcls_file) )
            error('PTCLS_FILE must be a char array.');
        end
        
        if( ~ischar(tomos_file) )
            if( isa(tomos_file,'SUSAN.Data.TomosInfo') )
                tomos_file.save([obj.work_dir 'tmp.tomostxt']);
                tomos_file = [obj.work_dir 'tmp.tomostxt'];
            else
                error('TOMOS_FILE must be a char array or a TOMOSINFO object.');
            end
        end
        
        gpu_list_str = sprintf('%d,',obj.gpu_list);
        gpu_list_str = gpu_list_str(1:end-1);
        
        num_threads = length(obj.gpu_list);
        
        susan_path = what('SUSAN');
        ctf_exec = [susan_path.path '/bin/estimate_ctf'];
        
        ctf_exec = [ctf_exec ' -tomos_in '    tomos_file];
        ctf_exec = [ctf_exec ' -data_out '    output_dir];
        ctf_exec = [ctf_exec ' -ptcls_file '  ptcls_file];
        ctf_exec = [ctf_exec ' -n_threads '   sprintf('%d',num_threads)];
        ctf_exec = [ctf_exec ' -binning '     sprintf('%d',obj.binning)];
        ctf_exec = [ctf_exec ' -gpu_list '    gpu_list_str];
        ctf_exec = [ctf_exec ' -box_size '    sprintf('%d',obj.box_size)];
        ctf_exec = [ctf_exec ' -res_range '   sprintf('%f,%f',obj.resolution.min,obj.resolution.max)];
        ctf_exec = [ctf_exec ' -def_range '   sprintf('%f,%f',obj.defocus.min,obj.defocus.max)];
        ctf_exec = [ctf_exec ' -tilt_search ' sprintf('%f',obj.tlt_range)];
        ctf_exec = [ctf_exec ' -refine_def '  sprintf('%f,%f',obj.refine_def.range,obj.refine_def.step)];
        ctf_exec = [ctf_exec ' -temp_dir '    obj.work_dir];
        
        stat = system(ctf_exec);
        
        if( stat == 0 )
            tomos = SUSAN.Data.TomosInfo.load(tomos_file);
            for i = 1:length(tomos.tomo_id)
                new_defocus_file = sprintf([output_dir 'tomo%03d/defocus.txt'],i);
                tomos.set_defocus(i,new_defocus_file)
            end
        else
            error('CtfEstimator crashed.');
        end
        
    end
    
end


end

