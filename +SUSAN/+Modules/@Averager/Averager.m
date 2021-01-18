classdef Averager < handle
% SUSAN.Modules.Averager Reconstruct volumes from substacks.
%    Reconstruct all the volumes from substacks from a ParticlesInfo file.
%
% SUSAN.Modules.Averager Properties:
%    gpu_list        - (vector) list of GPU IDs to use for alignment.
%    working_dir     - (string) working directory for temporarily files.
%    bandpass        - (struct) frequency range to use.
%    extra_padding   - (uint32) extra padding for the reconstruction.
%    inversion_iter  - (uint32) number of iterations for the gridding correction.
%    inversion_std   - (scalar) blob size for the gridding correction.
%    fourier_sigmod  - (scalar) tamper factor for the bandpass.
%    rec_halves      - (bool)   should reconstruct the half maps.
%    keep_temp_files - (bool)   delete the working directory when finish.
%    ssnr_f          - (single) SSNR function approximation. Denoising part.
%    ssnr_s          - (single) SSNR function approx. 0 = phaseflip, 1 = full inversion.
%    threads_per_gpu - (scalar) [experimental] multiple threads per GPU.
% 
% SUSAN.Modules.Averager Properties (Read-Only/set by methods):
%    ctf_correction_str - (string) type of CTF correction to be applied.
%
% SUSAN.Modules.Averager Methods:
%    set_ctf_disabled           - disables the CTF correction.
%    set_ctf_phase_flip         - uses traditional phase-flipping CTF correction.
%    set_ctf_inversion          - corrects the CTF using a Wiener filter.
%    set_ctf_inversion_bfactor  - corrects the CTF using a Wiener filter with the BFactor.
%    check_GPU                  - checks if the requested GPUs are available
%    reconstruct                - performs the reconstruction.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
properties
    gpu_list        uint32  = [];
    working_dir             = 'tmp/';
    bandpass                = struct('highpass',0,'lowpass',0);
    extra_padding   uint32  = 32;
    inversion_iter  uint32  = 10;
    inversion_std   single  = 0.5;
    fourier_sigmod  single  = 1.5;
    keep_temp_files logical = false;
    force_bfactor   single  = [];
    rec_halves      logical = false;
    ssnr_f          single  = 0;
    ssnr_s          single  = 0;
    threads_per_gpu uint32  = 1;
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
properties(SetAccess=private)
    ctf_correction_str = 'none';
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
properties(Access=private)
    ctf_correction_type uint32 = 0; % 0 = no correction
                                    % 1 = phase-flip
                                    % 2 = weights inversion (no BFactor)
                                    % 3 = weights inversion (with BFactor)
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

methods
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_ctf_disabled(obj)
    % SET_CTF_DISABLED disables the CTF correction.
    %   SET_CTF_DISABLED disables the CTF correction.
        obj.ctf_correction_type = 0;
        obj.ctf_correction_str  = 'none';
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_ctf_phase_flip(obj)
    % SET_CTF_PHASE_FLIP uses traditional phase-flipping CTF correction.
    %   SET_CTF_PHASE_FLIP uses traditional phase-flipping CTF correction.
        obj.ctf_correction_type = 1;
        obj.ctf_correction_str  = 'phase-flip';
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_ctf_inversion(obj)
    % SET_CTF_PHASE_FLIP corrects the CTF using a Wiener filter.
    %   SET_CTF_PHASE_FLIP corrects the CTF using a Wiener filter.
        obj.ctf_correction_type = 2;
        obj.ctf_correction_str  = 'inversion';
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_ctf_inversion_bfactor(obj)
    % SET_CTF_PHASE_FLIP corrects the CTF using a Wiener filter with the BFactor.
    %   SET_CTF_PHASE_FLIP corrects the CTF using a Wiener filter with the BFactor.
        obj.ctf_correction_type = 3;
        obj.ctf_correction_str  = 'bfactor-inversion';
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function rslt = check_GPU(obj)
    % CHECK_GPU checks if the requested GPUs are available.
    %   CHECK_GPU checks if the requested GPUs are available.
        
        if( obj.working_dir(end) ~= '/' )
            obj.working_dir = [obj.working_dir '/'];
        end
        
        obj.gpu_list = unique(obj.gpu_list);
        N = gpuDeviceCount;
        if( N < 1 )
            warning('No GPUs available.');
            rslt = false;
        else
            gpus_unavailable = obj.gpu_list( (obj.gpu_list+1) > N );
            if( isempty(gpus_unavailable) )
                rslt = true;
            else
                for i = 1:length(gpus_unavailable)
                    warning('Requested GPU %d is not available.',gpus_unavailable(i));
                end
                rslt = false;
            end
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function reconstruct(obj,out_prefix,tomo_list_file,particle_list_file,box_size)
    % RECONSTRUCT performs the reconstruction.
    %   RECONSTRUCT(OUT_PFX,TOMOS,PTCLS_IN,BOX_SIZE) reconstruct the volumnes
    %   described in PTCLS_IN, with a box size of BOX_SIZE, from the tomograms
    %   listed in TOMOS. The files are created using OUT_PFX as prefix. The
    %   number of the reference/class is appended to the prefix. If REC_HALVES
    %   is enabled, the half-sets are also reconstructed.
    %
    %   See also SUSAN.Data.TomosInfo and SUSAN.Data.ParticlesInfo
        
        if( ~ischar(tomo_list_file) )
            error('First argument (TOMO_LIST_FILE) must be a char array.');
        end
        
        if( ~ischar(particle_list_file) )
            error('Second argument (PARTICLE_LIST_FILE) must be a char array.');
        end
        
        if( ~obj.check_GPU() )
            error('Invalid GPU configuration');
        end
        
        if( ~exist(obj.working_dir,'dir') )
            mkdir(obj.working_dir);
        end
        
        if( ~SUSAN.Utils.is_extension(particle_list_file,'ptclsraw') )
            error(['File "' particle_list_file '" is not a SUSAN particle list file.']);
        end
        
        if( ~SUSAN.Utils.is_extension(tomo_list_file,'tomostxt') )
            error(['File "' tomo_list_file '" is not a SUSAN tomogram list file.']);
        end
        
        gpu_list_str = sprintf('%d,',obj.gpu_list);
        gpu_list_str = gpu_list_str(1:end-1);
        
        num_threads = length(obj.gpu_list)*obj.threads_per_gpu;
        
        if( obj.bandpass.lowpass <= obj.bandpass.highpass )
            obj.bandpass.lowpass = box_size/2;
        end
        R_range = single( sort( [obj.bandpass.highpass obj.bandpass.lowpass] ) );
        
        if( out_prefix(end) ~= '_' )
            out_prefix = [out_prefix '_'];
        end

        susan_path = what('SUSAN');
        averager_exec = [susan_path.path '/bin/averager'];
        
        averager_exec = [averager_exec ' -tomos_file ' tomo_list_file];
        averager_exec = [averager_exec ' -out_prefix ' out_prefix];
        averager_exec = [averager_exec ' -ptcls_file ' particle_list_file];
        averager_exec = [averager_exec ' -box_size '   sprintf('%d',box_size)];
        averager_exec = [averager_exec ' -padding '    sprintf('%d',obj.extra_padding)];
        averager_exec = [averager_exec ' -n_threads '  sprintf('%d',num_threads)];
        averager_exec = [averager_exec ' -gpu_list '   gpu_list_str];
        averager_exec = [averager_exec ' -ctf_type '   obj.ctf_correction_str];
        averager_exec = [averager_exec ' -w_inv_iter ' sprintf('%d',obj.inversion_iter)];
        averager_exec = [averager_exec ' -w_inv_gstd ' sprintf('%f',obj.inversion_std)];
        averager_exec = [averager_exec ' -rolloff_f '  sprintf('%f',obj.fourier_sigmod)];
        averager_exec = [averager_exec ' -bandpass '   sprintf('%f,%f',R_range(1),R_range(2))];
        averager_exec = [averager_exec ' -temp_dir '   obj.working_dir];
        averager_exec = [averager_exec ' -ssnr_f '     sprintf('%f',obj.ssnr_f)];
        averager_exec = [averager_exec ' -ssnr_s '     sprintf('%f',obj.ssnr_s)];
        
        if( obj.rec_halves > 0 )
            averager_exec = [averager_exec ' -rec_halves 1'];
        end
        
        if( ~isempty( obj.force_bfactor ) )
            averager_exec = [averager_exec ' -bfactor ' sprintf('%f',obj.force_bfactor)];
        end
        
        stat = system(averager_exec);
        
        if( ~obj.keep_temp_files )
            if( exist(obj.working_dir,'dir') )
                rmdir(obj.working_dir,'s');
            end
        end
        
        if(stat~=0)
            error('Averager crashed.');
        end
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

methods(Access=private)
    
    
end


end


