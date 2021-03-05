classdef Averager < handle
% SUSAN.Modules.Averager Reconstruct volumes from substacks.
%    Reconstruct all the volumes from substacks from a ParticlesInfo file.
%
% SUSAN.Modules.Averager Properties:
%    gpu_list        - (vector) list of GPU IDs to use for alignment.
%    bandpass        - (struct) frequency range to use.
%    padding         - (uint32) extra padding for the internal reconstruction.
%    inversion       - (struct) parameters for the W-matrix inversion.
%    rec_halves      - (bool)   should reconstruct the half maps.
%    threads_per_gpu - (scalar) [experimental] multiple threads per GPU.
% 
% SUSAN.Modules.Averager Properties (Read-Only/set by methods):
%    pad_type  - (string) Type of padding (zero/NOISE).
%    norm_type - (string) Type of normalization (none,ZERO_MEAN,zero_mean_proj_weight,zero_mean_one_std).
%    ctf_type  - (string) Type of CTF correction (none/phase_flip,WIENER,wiener_ssnr).
%    symmetry  - (string) Type of symmetry to be applied.
%    ssnr      - (struct) Params for the wiener_ssnr correction (~S*e^-F).
%
% SUSAN.Modules.Averager Methods:
%    set_ctf_correction - Sets the type of CTF correction: 
%                               - none.
%                               - phase_flip.
%                               - wiener.
%                               - wiener_ssnr.
%    set_padding_policy - Sets the padding policy:
%                               - zero.
%                               - noise.
%    set_normalization  - Sets the normalization policy:
%                               - none.
%                               - zero_mean.
%                               - zero_mean_one_std.
%                               - zero_mean_proj_weight.
%    set_symmetry       - Sets the symmetry type:
%                               - cX (c1,c2,c3,c4,....).
%                               - cbo (cuboctahedral).
%    reconstruct        - performs the reconstruction.
%    show_cmd           - show the command to execute the reconstruction.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
properties
    gpu_list        uint32  = [];
    bandpass                = struct('highpass',0,'lowpass',0,'rolloff',2);
    padding         uint32  = 32;
    inversion               = struct('iter',10,'gstd',0.5);
    rec_halves      logical = false;
    threads_per_gpu uint32  = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
properties(SetAccess=private)
    pad_type  = 'noise';
    norm_type = 'zero_mean';
    ctf_type  = 'wiener';
    symmetry  = 'c1';
    ssnr      = struct('F',0,'S',1);
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

methods
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_ctf_correction(obj,type,ssnr_s,ssnr_f)
    % SET_CTF_CORRECTION configures the type of CTF correction to be used.
    %   SET_CTF_CORRECTION(TYPE) configures the type of CTF correction.
    %   Supported values are:
    %     'none'       : Disables the ctf correction.
    %     'phase_flip' : Simple phase-flip correction.
    %     'wiener'     : Wiener inversion of the CTF: (CTF_i*P_i)/(CTF_i)^2
    %     'wiener_ssnr': Wiener inversion with parametrized SSNR.
    %   SET_CTF_CORRECTION('wiener_ssnr',S,F) configures the CTF correction
    %   as a Wiener inversion with the S and F parameters:
    %                          CTF_i*P_i
    %                  ------------------------,
    %                  (CTF_i)^2 + SSNR(f)^(-1)
    %   where SSNR(f) = (10^(3*S)) * exp( -f * 100*F ).
    %   Default values: S=1, F=0
    
    
        if( nargin < 4 )
            ssnr_f = 0;
        end
        
        if( nargin < 3 )
            ssnr_s = 1;
        end
        
        if( strcmpi(type,'none') )
            obj.ctf_type = 'none';
        elseif( strcmpi(type,'phase_flip') )
            obj.ctf_type = 'phase_flip';
        elseif( strcmpi(type,'wiener') )
            obj.ctf_type = 'wiener';
        elseif( strcmp(type,'wiener_ssnr') )
            obj.ctf_type = 'wiener_ssnr';
            obj.ssnr.S = ssnr_s;
            obj.ssnr.F = ssnr_f;
        else
            error('Invalid correction type. Accepted values: none, phase_flip, wiener, and wiener_ssnr');
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_padding_policy(obj,type)
    % SET_PADDING_POLICY sets the type of data used to fill the extra padding.
    %   SET_PADDING_POLICY(TYPE) supported values are:
    %     'zero'  : Fills the padding with 0.
    %     'noise' : Fills the padding with a gaussian distribution with the
    %               same mean and standard deviation as each projection.
        
        if( strcmpi(type,'zero') )
            obj.pad_type = 'zero';
        elseif( strcmpi(type,'noise') )
            obj.pad_type = 'noise';
        else
            error('Invalid padding policy. Accepted values: zero and noise');
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_normalization(obj,type)
    % SET_NORMALIZATION sets the normalization policy of each projection.
    %   SET_NORMALIZATION(TYPE) supported values are:
    %     'none' : Disables the normalization.
    %     'zm'   : Normalizes each projection to have mean=0
    %     'zm1s' : Normalizes each projection to have mean=0 and std=1
    %     'zmws' : Normalizes each projection to have mean=0 and std=W,
    %              where W is read from the prj_w attribure of each
    %              projection of the particle.
        obj.norm_type = type;

        if( strcmpi(type,'none') )
            obj.norm_type = 'none';
        elseif( strcmpi(type,'zm') )
            obj.norm_type = 'zero_mean';
        elseif( strcmpi(type,'zm1s') )
            obj.norm_type = 'zero_mean_one_std';
        elseif( strcmp(type,'zmws') )
            obj.norm_type = 'zero_mean_proj_weight';
        else
            error('Invalid normalization type. Accepted values: none, zm, zm1s, and zmws');
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_symmetry(obj,type)
    % SET_SYMMETRY sets the symmetry to be applied.
    %   SET_SYMMETRY(TYPE) supported values are:
    %     'none' : No symmetry.
    %     'cX'   : Copies along Z-axis. X is the number of copies.
    %     'cbo'  : Cuboctahedral symmetry.
        obj.symmetry = 'c1';

        if( strcmpi(type,'none') )
            obj.symmetry = 'c1';
        elseif( strcmpi(type,'cbo') )
            obj.symmetry = 'cbo';
        elseif( lower(type(1)) == 'c' && ~isnan(str2double(type(2:end))) )
            obj.symmetry = ['c' type(2:end)];
        else
            error('Invalid symmetry type. Accepted values: none, cbo, and cXX');
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
        
        rec_exec = obj.show_cmd(out_prefix,tomo_list_file,particle_list_file,box_size);
        
        stat = system(rec_exec);
        
        if(stat~=0)
            error('Averager crashed.');
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function cmd = show_cmd(obj,out_prefix,tomo_list_file,particle_list_file,box_size)
    % SHOW_CMD returns the command to execute the reconstruction.
    %   CMD = SHOW_CMD(OUT_PFX,TOMOS,PTCLS_IN,BOX_SIZE) command to execute 
    %   the reconstruction.
        
        if( ~ischar(tomo_list_file) )
            error('First argument (TOMO_LIST_FILE) must be a char array.');
        end
        
        if( ~ischar(particle_list_file) )
            error('Second argument (PARTICLE_LIST_FILE) must be a char array.');
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
        
        if( out_prefix(end) == '_' )
            out_prefix = out_prefix(1:end-1);
        end

        cmd = [SUSAN.bin_path '/susan_reconstruct'];   
        cmd = [cmd ' -tomos_file ' tomo_list_file];
        cmd = [cmd ' -out_prefix ' out_prefix];
        cmd = [cmd ' -ptcls_file ' particle_list_file];
        cmd = [cmd ' -n_threads '  sprintf('%d',num_threads)];
        cmd = [cmd ' -gpu_list '   gpu_list_str];
        cmd = [cmd ' -box_size '   sprintf('%d',box_size)];
        cmd = [cmd ' -pad_size '   sprintf('%d',obj.padding)];
        cmd = [cmd ' -pad_type '   obj.pad_type];
        cmd = [cmd ' -norm_type '  obj.norm_type];
        cmd = [cmd ' -ctf_type '   obj.ctf_type];
        cmd = [cmd ' -ssnr_param ' sprintf('%f,%f',obj.ssnr.F,obj.ssnr.S)];
        cmd = [cmd ' -w_inv_iter ' sprintf('%d',obj.inversion.iter)];
        cmd = [cmd ' -w_inv_gstd ' sprintf('%f',obj.inversion.gstd)];
        cmd = [cmd ' -bandpass '   sprintf('%f,%f',R_range(1),R_range(2))];
        cmd = [cmd ' -rolloff_f '  sprintf('%f',obj.bandpass.rolloff)];
        cmd = [cmd ' -symmetry '   obj.symmetry];
        
        if( obj.rec_halves > 0 )
            cmd = [cmd ' -rec_halves 1'];
        end
        
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

methods(Access=private)
    
    
end


end


