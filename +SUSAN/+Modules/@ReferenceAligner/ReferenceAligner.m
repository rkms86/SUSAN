classdef ReferenceAligner < handle
% SUSAN.Modules.ReferenceAligner Aligns halfmaps
%    Performs the 3D alignment of halfmaps
%
% SUSAN.Modules.ReferenceAligner Properties:
%    gpu_id   - (uint32) GPU ID to use for alignment.
%    bandpass - (struct) frequency range to use.
%    pix_size - (single) Pixel size.
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
    gpu_id   uint32 = 0;
    bandpass        = struct('highpass',0,'lowpass',0,'rolloff',3);
    pix_size single = -1;
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
properties(SetAccess=private)
    cone = struct('range',360,'step',30);
    inplane = struct('range',360,'step',30);
    refine = struct('level',2,'factor',2);
    offset = struct('range',[10, 10, 10],'step',1,'type','ellipsoid');
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
    %   - 2 elements: the ellipsoid will have the first element of the
    %                 range for the X and Y range, and the second element
    %                 defines the Z range.
    %   - 3 elements: each element defines the range in each axis.
    %   The OFFSET_STEP defines how the OFFSET_RANGE will be sampled. This
    %   defines the speed of the algorithm, a finer step leads to a slower
    %   execution. Note: a large step size (>1) is recommended in combination to
    %   lower frequancies for large OFFSET_RANGE. A smaller step size (<1)
    %   is used in combination if higher frequencies.
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
    %   - 2 elements: the first element defines the radius, the second
    %                 defines the semi-height.
    %   - 3 elements: each element defines the range in each axis.
    %   The OFFSET_STEP defines how the OFFSET_RANGE will be sampled. This
    %   defines the speed of the algorithm, a finer step leads to a slower
    %   execution. Note: a large step size (>1) is recommended in combination to
    %   lower frequancies for large OFFSET_RANGE. A smaller step size (<1)
    %   is used in combination if higher frequencies.
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
    function align(obj,ptcls_out,refs_list,ptcls_in,box_size)
    % ALIGN performs the alignment.
    %   ALIGN(PTCLS_OUT,REFS,PTCLS_IN,BOX_SIZE) aligns the halfmaps listed
    %   in the REFS file and updates the alignment values in PTCLS_IN and
    %   stores them in PTCLS_OUT. The box size of the halfmaps must be
    %   BOX_SIZE.
        
        ali_exec = obj.show_cmd(ptcls_out,refs_list,ptcls_in,box_size);
        
        stat = system(ali_exec);
        
        if(stat~=0)
            error('Refserence aligner crashed.');
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function cmd = show_cmd(obj,ptcls_out,refs_list,ptcls_in,box_size)
    % SHOW_CMD returns the command to execute the alignment.
    %   CMD = SHOW_CMD(PTCLS_OUT,REFS,PTCLS_IN,BOX_SIZE) command to execute 
    %   the alignment.
        
        if( ~ischar(refs_list) )
            error('Second argument (REFS) must be a char array.');
        end
        
        if( ~ischar(ptcls_in) )
            error('Third argument (PARTICLE_LIST_FILE) must be a char array.');
        end
        
        if( ~ischar(ptcls_out) )
            error('First argument (PARTICLE_LIST_FILE) must be a char array.');
        end
        
        if( ~SUSAN.Utils.is_extension(ptcls_in,'ptclsraw') )
            error(['File "' ptcls_in '" is not a SUSAN particle list file.']);
        end
        
        if( ~SUSAN.Utils.is_extension(refs_list,'refstxt') )
            error(['File "' refs_list '" is not a SUSAN tomogram list file.']);
        end
        
        if( obj.bandpass.lowpass <= obj.bandpass.highpass )
            obj.bandpass.lowpass = box_size/2;
        end
        R_range = single( sort( [obj.bandpass.highpass obj.bandpass.lowpass] ) );
        
        cmd = [SUSAN.bin_path '/susan_refs_aligner'];   
        cmd = [cmd ' -ptcls_in '   ptcls_in];
        cmd = [cmd ' -ptcls_out '  ptcls_out];
        cmd = [cmd ' -refs_file '  refs_list];
        cmd = [cmd ' -gpu_id '     sprintf('%d',obj.gpu_id)];
        cmd = [cmd ' -box_size '   sprintf('%d',box_size)];
        cmd = [cmd ' -bandpass '   sprintf('%f,%f',R_range(1),R_range(2))];
        cmd = [cmd ' -rolloff_f '  sprintf('%f',obj.bandpass.rolloff)];
        offset_str = sprintf('%f,',obj.offset.range);
        offset_str = [offset_str sprintf('%f',obj.offset.step)];
        cmd = [cmd ' -cone '       sprintf('%f,%f',obj.cone.range,obj.cone.step)];
        cmd = [cmd ' -inplane '    sprintf('%f,%f',obj.inplane.range,obj.inplane.step)];
        cmd = [cmd ' -refine '     sprintf('%d,%d',obj.refine.factor,obj.refine.level)];
        cmd = [cmd ' -off_type '   obj.offset.type];
        cmd = [cmd ' -off_params ' offset_str];
        
        if( obj.pix_size > 0 )
            cmd = [cmd ' -pix_size ' sprintf('%f',obj.pix_size)];
        end
        
    end
end


end


