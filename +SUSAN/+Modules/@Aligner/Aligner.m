classdef Aligner < handle
% SUSAN.Modules.Aligner Performs the 3D/2D alignment.
%    Performs the 3D/2D alignment.
%
% SUSAN.Modules.Aligner Properties:
%    gpu_list        - (vector) list of GPU IDs to use for alignment.
%    bandpass        - (struct) frequency range to use.
%    type            - (int)    dimensionality of the alignment (3 or 2).
%    drift           - (bool)   allow particle to drift.
%    halfsets        - (bool)   align independent halfsets.
%    threads_per_gpu - (scalar) multiple threads per GPU.
% 
% SUSAN.Modules.Aligner Properties (Read-Only/set by methods):
%    cone    - (struct) cone angles search range/step
%    inplane - (struct) inplane angles search range/step
%    refine  - (struct) refine angles levels/factor
%    offset  - (struct) offset range/step/type
%
% SUSAN.Modules.Aligner Methods:
%    set_angular_search     - configures the angular search (cone and inplane)
%    set_angular_refinement - configures the refinement levels/factor.
%    set_offset_ellipsoid   - configures the offset range as an ellipsoid.
%    set_offset_cylinder    - configures the offset range as a cylinder.
%    align                  - performs the alignment.
%    show_cmd               - show the command to execute the alignment.

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
    type            uint32  = 3;
    padding         uint32  = 32;
    drift           logical = false;
    halfsets        logical = false;
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
properties(SetAccess=private)
    cone = struct('range',360,'step',30);
    inplane = struct('range',360,'step',30);
    refine = struct('level',2,'factor',2);
    offset = struct('range',[10, 10, 10],'step',1,'type','ellipsoid');
    pad_type  = 'noise';
    norm_type = 'zero_mean';
    ctf_type  = 'substack';
    symmetry  = 'c1';
    cc_type   = 'basic';
    ssnr      = struct('F',0,'S',1);
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
properties(Access=private)
    
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

methods
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_angular_search(obj,cone_range,cone_step,inplane_range,inplane_step)
    % SET_ANGULAR_SEARCH configures the angular search (cone and inplane).
    %   SET_ANGULAR_SEARCH(CONE_RANGE,CONE_STEP,INPLANE_RANGE,INPLANE_STEP)
    %   configures the angular search grid similar to DYNAMO.
    %
    %   See also https://wiki.dynamo.biozentrum.unibas.ch/w/index.php/Multilevel_refinement
        obj.cone.range    = cone_range;
        obj.cone.step     = cone_step;
        obj.inplane.range = inplane_range;
        obj.inplane.step  = inplane_step;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_angular_refinement(obj,refine_level,refine_factor)
    % SET_ANGULAR_REFINEMENT configures the refinement levels/factor.
    %   SET_ANGULAR_REFINEMENT(REFINE_LEVEL,REFINE_FACTOR) configures the
    %   angular refinement in an equivalent way as DYNAMO.
    %
    %   See also https://wiki.dynamo.biozentrum.unibas.ch/w/index.php/Multilevel_refinement
        obj.refine.level  = refine_level;
        obj.refine.factor = refine_factor;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_offset_ellipsoid(obj,offset_range,offset_step)
    % SET_OFFSET_ELLIPSOID configures the offset range as an ellipsoid.
    %   SET_OFFSET_ELLIPSOID(OFFSET_RANGE,OFFSET_STEP) sets the offset
    %   search range between the reference and the particles. The number of
    %   elements in OFFSET_RANGE defines the shape of the ellipsoid:
    %   - 1 element:  the ellipsoid becomes an sphere.
    %                 Ex.: SET_OFFSET_ELLIPSOID(r);
    %   - 2 elements: the ellipsoid will have the first element of the
    %                 range for the X and Y range, and the second element
    %                 defines the Z range.
    %                 Ex.: SET_OFFSET_ELLIPSOID([r,r_z]);
    %   - 3 elements: each element defines the range in each axis.
    %                 Ex.: SET_OFFSET_ELLIPSOID([r_x,r_y,r_z]);
    %   The OFFSET_STEP defines how the OFFSET_RANGE will be sampled. This
    %   defines the speed of the algorithm, a finer step leads to a slower
    %   execution. Note: a large step size (>1) is recommended in combination to
    %   lower frequancies for large OFFSET_RANGE. A smaller step size (<1)
    %   is used in combination if higher frequencies.
    %     Ex.: SET_OFFSET_ELLIPSOID([r_x,r_y,r_z],subsample);
    %
    %   See also set_offset_ellipsoid.
        
        if( nargin < 3 )
            offset_step = 1;
        end
        
        if( length(offset_range) == 1 )
            obj.offset.range = [offset_range offset_range offset_range];
        elseif( length(offset_range) == 2 )
            obj.offset.range = [offset_range(1) offset_range(1) offset_range(2)];
        elseif( length(offset_range) == 3 )
            obj.offset.range = offset_range;
        else
            error('Invalid length of offset_range');
        end
        
        obj.offset.step  = offset_step;
        obj.offset.type  = 'ellipsoid';
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_offset_cylinder(obj,offset_range,offset_step)
    % SET_OFFSET_CYLINDER configures the offset range as a cylinder.
    %   SET_OFFSET_CYLINDER(OFFSET_RANGE,OFFSET_STEP) sets the offset
    %   search range between the reference and the particles. The number of
    %   elements in OFFSET_RANGE defines the shape of the cylinder:
    %   - 1 element:  the height is the same as its diameter.
    %                 Ex.: SET_OFFSET_CYLINDER(r);
    %   - 2 elements: the first element defines the radius, the second
    %                 defines the semi-height.
    %                 Ex.: SET_OFFSET_CYLINDER([r,h]);
    %   - 3 elements: each element defines the range in each axis.
    %                 Ex.: SET_OFFSET_CYLINDER([r_x,r_y,h]);
    %   The OFFSET_STEP defines how the OFFSET_RANGE will be sampled. This
    %   defines the speed of the algorithm, a finer step leads to a slower
    %   execution. Note: a large step size (>1) is recommended in combination to
    %   lower frequancies for large OFFSET_RANGE. A smaller step size (<1)
    %   is used in combination if higher frequencies.
    %     Ex.: SET_OFFSET_CYLINDER([r_x,r_y,r_z],subsample);
    %
    %   See also set_offset_cylinder.
        
        if( nargin < 3 )
            offset_step = 1;
        end
        
        if( length(offset_range) == 1 )
            obj.offset.range = [offset_range offset_range offset_range];
        elseif( length(offset_range) == 2 )
            obj.offset.range = [offset_range(1) offset_range(1) offset_range(2)];
        elseif( length(offset_range) == 3 )
            obj.offset.range = offset_range;
        else
            error('Invalid length of offset_range');
        end
        
        obj.offset.step  = offset_step;
        obj.offset.type  = 'cylinder';
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_ctf_correction(obj,type,ssnr_s,ssnr_f)
    % SET_CTF_CORRECTION configures the type of CTF correction to be used.
    %   SET_CTF_CORRECTION(TYPE) configures the type of CTF correction.
    %   Supported values are:
    %     'none'         : Disables the ctf correction.
    %     'on_reference' : Apply CTF on the reference.
    %     'on_substack'  : Wiener inversion of the CTF: (CTF_i*P_i)/(CTF_i)^2
    %     'wiener_ssnr'  : Wiener inversion with parametrized SSNR.
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
        elseif( strcmpi(type,'on_reference') )
            obj.ctf_type = 'on_reference';
        elseif( strcmpi(type,'on_substack') )
            obj.ctf_type = 'on_substack';
        elseif( strcmp(type,'wiener_ssnr') )
            obj.ctf_type = 'wiener_ssnr';
            obj.ssnr.S = ssnr_s;
            obj.ssnr.F = ssnr_f;
        elseif( strcmpi(type,'wiener_white') )
            obj.ctf_type = 'wiener_white';
        elseif( strcmpi(type,'wiener_phase') )
            obj.ctf_type = 'wiener_phase';
        elseif( strcmpi(type,'cfsc') )
            obj.ctf_type = 'cfsc';
        else
            error('Invalid correction type. Accepted values: none, on_reference, on_substack, and wiener_ssnr');
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
    function set_pseudosymmetry(obj,type)
    % SET_PSEUDOSYMMETRY sets the symmetry to be applied.
    %   SET_PSEUDOSYMMETRY(TYPE) supported values are:
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
            error('Invalid pseudo-symmetry type. Accepted values: none, cbo, and cXX');
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function align(obj,ptcls_out,refs_list,tomo_list,ptcls_in,box_size)
    % ALIGN performs the alignment.
    %   ALIGN(PTCLS_OUT,REFS,TOMOS,PTCLS_IN,BOX_SIZE) aligns the particles 
    %   PTCLS_IN to the references in REFS, using the tomograms' information 
    %   in TOMOS, and saving the results in PTLCS_OUT.
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
        
        ali_exec = obj.show_cmd(ptcls_out,refs_list,tomo_list,ptcls_in,box_size);
        
        stat = system(ali_exec);
        
        if(stat~=0)
            error('Aligner crashed.');
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function cmd = show_cmd(obj,ptcls_out,refs_list,tomo_list,ptcls_in,box_size)
    % SHOW_CMD returns the command to execute the alignment.
    %   CMD = SHOW_CMD(PTCLS_OUT,REFS,TOMOS,PTCLS_IN,BOX_SIZE) command to execute 
    %   the alignment.
        
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
            cmd = 'susan_aligner';
        else
            cmd = [SUSAN.bin_path '/susan_aligner'];
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
        cmd = [cmd ' -norm_type '  obj.norm_type];
        cmd = [cmd ' -ctf_type '   obj.ctf_type];
        cmd = [cmd ' -ssnr_param ' sprintf('%f,%f',obj.ssnr.F,obj.ssnr.S)];
        cmd = [cmd ' -bandpass '   sprintf('%f,%f',R_range(1),R_range(2))];
        cmd = [cmd ' -rolloff_f '  sprintf('%f',obj.bandpass.rolloff)];
        cmd = [cmd ' -p_symmetry ' obj.symmetry];
        cmd = [cmd ' -cc_type '    obj.cc_type];

        if( obj.halfsets > 0 )
            cmd = [cmd ' -ali_halves 1'];
        else
            cmd = [cmd ' -ali_halves 0'];
        end
        
        if( obj.drift > 0 )
            cmd = [cmd ' -allow_drift 1'];
        else
            cmd = [cmd ' -allow_drift 0'];
        end
        
        offset_str = sprintf('%f,',obj.offset.range);
        offset_str = [offset_str sprintf('%f',obj.offset.step)];
        cmd = [cmd ' -cone '       sprintf('%f,%f',obj.cone.range,obj.cone.step)];
        cmd = [cmd ' -inplane '    sprintf('%f,%f',obj.inplane.range,obj.inplane.step)];
        cmd = [cmd ' -refine '     sprintf('%d,%d',obj.refine.factor,obj.refine.level)];
        cmd = [cmd ' -off_type '   obj.offset.type];
        cmd = [cmd ' -off_params ' offset_str];
        
        if( obj.type == 3 )
            cmd = [cmd ' -type 3'];
        elseif( obj.type == 2 )
            cmd = [cmd ' -type 2'];
        else
            error('Invalid alignment type');
        end
        
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end


