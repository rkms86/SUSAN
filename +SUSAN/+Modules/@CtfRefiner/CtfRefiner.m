classdef CtfRefiner < handle
% SUSAN.Modules.CtfRefiner Performs per-particle-per-projection CTF refinement.
%    Performs per-particle-per-projection CTF refinement.
%
% SUSAN.Modules.CtfRefiner Properties:
%    gpu_list        - (vector) list of GPU IDs to use for alignment.
%    bandpass        - (struct) frequency range to use.
%    halfsets        - (bool)   refine using independent halfsets.
%    threads_per_gpu - (scalar) multiple threads per GPU.
%    defocus         - (struct) search range/step.
%    angles          - (struct) search range/step.
%    estimate_dose_w - (bool)   enabling dose weight estimation.
%    
% SUSAN.Modules.CtfRefiner Methods:
%    refine   - performs the refinement.
%    show_cmd - show the command to execute the refinement.

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
properties
    gpu_list        uint32  = [];
    threads_per_gpu uint32  = 1;
    bandpass                = struct('highpass',0,'lowpass',0,'rolloff',3);
    padding         uint32  = 32;
    defocus                 = struct('range',1000,'step',100);
    angles                  = struct('range',10,'step',2);
    estimate_dose_w logical = false;
    halfsets        logical = false;
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
properties(SetAccess=private)
    pad_type  = 'noise';
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
properties(Access=private)
    
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

methods
    
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
    function refine(obj,ptcls_out,refs_list,tomo_list,ptcls_in,box_size)
    % REFINE performs the refinement.
    %   REFINE(PTCLS_OUT,REFS,TOMOS,PTCLS_IN,BOX_SIZE) refines the defocus
    %   values of the particles PTCLS_IN in regards to the references in 
    %   REFS, using the tomograms' information in TOMOS, and saving the 
    %   results in PTLCS_OUT.
    %   - PTCLS_OUT is a file where the updated information of the
    %     particles will be stored.
    %   - REFS is a filename of a stored SUSAN.Data.ReferenceInfo object.
    %   - TOMOS is a filename of a stored SUSAN.Data.TomosInfo object.
    %   - PTCLS_IN is a filename of a stored SUSAN.Data.ParticlesInfo
    %     object holding the information of the particles to be aligned. It
    %     must match the information in TOMOS and REFS.
    %   - BOX_SIZE self-explanatory.
    %
    %   See also SUSAN.Data.ReferenceInfo, SUSAN.Data.TomosInfo and SUSAN.Data.ParticlesInfo
        
        ref_exec = obj.show_cmd(ptcls_out,refs_list,tomo_list,ptcls_in,box_size);
        
        stat = system(ref_exec);
        
        if(stat~=0)
            error('Ctf Refiner crashed.');
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function cmd = show_cmd(obj,ptcls_out,refs_list,tomo_list,ptcls_in,box_size)
    % SHOW_CMD returns the command to execute the refinement.
    %   CMD = SHOW_CMD(PTCLS_OUT,REFS,TOMOS,PTCLS_IN,BOX_SIZE) command to execute 
    %   the refinement.
        
        if( ~ischar(tomo_list) )
            error('Third argument (TOMOS) must be a char array.');
        end
        
        if( ~ischar(refs_list) )
            error('Second argument (REFS) must be a char array.');
        end
        
        if( ~ischar(ptcls_in) )
            error('Forth argument (PARTICLE_LIST_FILE) must be a char array.');
        end
        
        if( ~ischar(ptcls_out) )
            error('First argument (PARTICLE_LIST_FILE) must be a char array.');
        end
        
        if( ~SUSAN.Utils.is_extension(ptcls_in,'ptclsraw') )
            error(['File "' ptcls_in '" is not a SUSAN particle list file.']);
        end
        
        if( ~SUSAN.Utils.is_extension(tomo_list,'tomostxt') )
            error(['File "' tomo_list_file '" is not a SUSAN tomogram list file.']);
        end
        
        if( ~SUSAN.Utils.is_extension(refs_list,'refstxt') )
            error(['File "' refs_list '" is not a SUSAN tomogram list file.']);
        end
        
        gpu_list_str = sprintf('%d,',obj.gpu_list);
        gpu_list_str = gpu_list_str(1:end-1);
        
        num_threads = length(obj.gpu_list)*obj.threads_per_gpu;
        
        if( obj.bandpass.lowpass <= obj.bandpass.highpass )
            obj.bandpass.lowpass = box_size/2;
        end
        R_range = single( sort( [obj.bandpass.highpass obj.bandpass.lowpass] ) );
        
        if( isdeployed )
            cmd = 'susan_refine_ctf';
        else
            cmd = [SUSAN.bin_path '/susan_refine_ctf'];
        end
        cmd = [cmd ' -tomos_file ' tomo_list];
        cmd = [cmd ' -ptcls_in '   ptcls_in];
        cmd = [cmd ' -ptcls_out '  ptcls_out];
        cmd = [cmd ' -refs_file '  refs_list];
        cmd = [cmd ' -n_threads '  sprintf('%d',num_threads)];
        cmd = [cmd ' -gpu_list '   gpu_list_str];
        cmd = [cmd ' -box_size '   sprintf('%d',box_size)];
        cmd = [cmd ' -pad_size '   sprintf('%d',obj.padding)];
        cmd = [cmd ' -pad_type '   obj.pad_type];
        cmd = [cmd ' -bandpass '   sprintf('%f,%f',R_range(1),R_range(2))];
        cmd = [cmd ' -rolloff_f '  sprintf('%f',obj.bandpass.rolloff)];
        
        cmd = [cmd ' -def_search '  sprintf('%f,%f',obj.defocus.range,obj.defocus.step)];
        cmd = [cmd ' -ang_search '  sprintf('%f,%f',obj.angles.range,obj.angles.step)];
        
        if( obj.halfsets > 0 )
            cmd = [cmd ' -use_halves 1'];
        else
            cmd = [cmd ' -use_halves 0'];
        end
        
        if( obj.estimate_dose_w > 0 )
            cmd = [cmd ' -est_dose 1'];
        else
            cmd = [cmd ' -est_dose 0'];
        end
        
    end
end

end


