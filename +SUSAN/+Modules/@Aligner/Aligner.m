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

%
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     function show_angles_cone(obj)
%     % SHOW_ANGLES_CONE shows the configured angles to search (cone).
%     %   SHOW_ANGLES_CONE creates a figure that shows the angular grid that
%     %   will be used for the alignment process (cone search values).
%     %
%     %   See also set_angular_search and set_angular_refinement.
%         figure;
%         obj.show_angles_cone_primitive();
%     end
%     
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     function show_angles_inplane(obj)
%     % SHOW_ANGLES_INPLANE shows the configured angles to search (inplane).
%     %   SHOW_ANGLES_INPLANE creates a figure that shows the angular grid 
%     %   that will be used for the alignment process (inplane search values).
%     %
%     %   See also set_angular_search and set_angular_refinement.
%         figure;
%         obj.show_angles_inplane_primitive();
%     end
%     
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     function show_angles(obj)
%     % SHOW_ANGLES shows the configured angles to search (cone and inplane).
%     %   SHOW_ANGLES creates a figure that plots two subfigures, one with
%     %   the angular grid of the cone search, and the other one with the
%     %   inplane search information.
%     %
%     %   See also show_angles_cone, show_angles_inplane, set_angular_search and set_angular_refinement.
%         figure;
%         subplot(1,2,1);
%         obj.show_angles_cone_primitive();
%         subplot(1,2,2);
%         obj.show_angles_inplane_primitive();
%     end
%     
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     function show_offset(obj)
%     % SHOW_OFFSET shows the search offset range.
%     %   SHOW_OFFSET creates a figure that shows the points where the cross
%     %   correlation will be sampled. The location of the cross correlation
%     %   peak defines the offset between the reference and the particle.
%     %
%     %   See also set_offset_ellipsoid and set_offset_cylinder.
%         figure;
%         obj.show_points_primitive();
%     end
%     
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     function show_angles_and_offset(obj)
%     % SHOW_ANGLES_AND_OFFSET shows the angles and offset search values.
%     %   SHOW_ANGLES creates a figure that plots three subfigures, one with
%     %   the angular grid of the cone search, the next one with the inplane 
%     %   search information, and the last one with the offset grid.
%     %
%     %   See also show_angles and show_offset.
%         figure;
%         subplot(1,3,1);
%         obj.show_angles_cone_primitive();
%         subplot(1,3,2);
%         obj.show_angles_inplane_primitive();
%         subplot(1,3,3);
%         obj.show_points_primitive();
% %         position = get(gcf,'Position');
% %         position(3:4) = [720 440];
% %         set(gcf,'Position',position);
%     end
    
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
            obj.norm_type = 'zero_mean_one_weight';
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
        
        cmd = [SUSAN.bin_path '/susan_aligner'];   
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
        cmd = [cmd ' -refine '     sprintf('%d,%d',obj.refine.level,obj.refine.factor)];
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

methods(Access=private)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function show_angles_cone_primitive(obj)

        cr  = obj.cone_range;
        cs  = obj.cone_step;
        rl  = obj.refine_level;
        rf  = obj.refine_factor;
        sym = 'c1';
        
        c = lines;
        
        [pts,lvl] = Aligner_list_cone_search(cr,cs,rl,rf,sym);
        if( ~isempty(lvl) )
            
            hold on;
            [X,Y,Z] = sphere(15);
            mesh(0.95*X,0.95*Y,0.95*Z,'EdgeColor',[0.1 0.1 0.1],'HandleVisibility','off');
            
            lvls = unique(lvl);
            
            legend_text = cell(length(lvls),1);
            
            for lvl_it = lvls
                
                lvl_pts = pts( lvl == lvl_it, : );
                
                plot3(lvl_pts(:,1),lvl_pts(:,2),lvl_pts(:,3),'.','MarkerSize',12,'Color',c(lvl_it,:));
                
                if( lvl_it > 1 )
                    legend_text{lvl_it} = sprintf('Refinements level %2d',lvl_it-1);
                else
                    legend_text{1} = 'Initial angular search.';
                end
                
            end
            
            lgnd = legend(legend_text,'Location','southoutside');
            set(lgnd, 'FontName', 'DejaVu Sans Mono');
            xticklabels('');
            yticklabels('');
            zticklabels('');
            grid on;
            axis equal;
            max_length = (1+single(max(lvls)+1)*0.1);
            axis([-max_length max_length -max_length max_length -max_length max_length]);
            title('Angular search - Cone');
            hold off;
            
            view(-37.5,30);
            
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function show_angles_inplane_primitive(obj)

        ir  = obj.inplane_range;
        is  = obj.inplane_step;
        rl  = obj.refine_level;
        rf  = obj.refine_factor;
        sym = 'c1';
        
        c = lines;
        
        [pts_x,pts_y,lvl] = Aligner_list_inplane_search(ir,is,rl,rf,sym);
        if( ~isempty(lvl) )
            
            hold on;
            
            lvls = unique(lvl);
            
            h = zeros(length(lvls), 1);
            legend_text = cell(length(lvls),1);
            
            for lvl_it = lvls
                
                plot(pts_x(lvl==lvl_it,:)',pts_y(lvl==lvl_it,:)','.-','MarkerSize',12,'Color',c(lvl_it,:));
                
                if( lvl_it > 1 )
                    legend_text{lvl_it} = sprintf('Refinements level %2d',lvl_it-1);
                else
                    legend_text{1} = 'Initial angular search.';
                end
                
                h(lvl_it) = plot(NaN,NaN,'.-','MarkerSize',12,'Color',c(lvl_it,:));
            end
            
            lgnd = legend(h,legend_text,'Location','southoutside');
            set(lgnd, 'FontName', 'DejaVu Sans Mono');
            xticklabels('');
            yticklabels('');
            grid on;
            axis equal;
            max_length = (1+single(max(lvls)+1)*0.1);
            axis([-max_length max_length -max_length max_length]);
            title('Angular search - Inplane');
            hold off;
            
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function show_points_primitive(obj)
        if( strcmp(obj.offset_type,'ellipsoid') )
            P = Aligner_list_points_ellipsoid(obj.offset_range,obj.offset_step);
        elseif( strcmp(obj.offset_type,'cylinder') )
            P = Aligner_list_points_cylinder(obj.offset_range,obj.offset_step);
        else
            error('Unknown offset type');
        end
        
        if( ~isempty(P) )
            
            x_lim = max(P(1,:)) + obj.offset_step;
            y_lim = max(P(2,:)) + obj.offset_step;
            z_lim = max(P(3,:)) + obj.offset_step;
            
            plot3(P(1,:),P(2,:),P(3,:),'.');
            grid on;
            axis equal;
            axis([-x_lim x_lim -y_lim y_lim -z_lim z_lim]);
            lgnd = legend(sprintf('Offset type: %s',obj.offset_type),'Location','southoutside');
            set(lgnd, 'FontName', 'DejaVu Sans Mono');
            xlabel('X');
            ylabel('Y');
            zlabel('Z');
%             xticks(-x_lim:x_lim);
%             yticks(-y_lim:y_lim);
%             zticks(-z_lim:z_lim);
            hold off;
            ax = gca;
            ax.XMinorGrid = 'on';
            ax.XAxis.MinorTickValues = -x_lim:obj.offset_step:x_lim;
            ax.YMinorGrid = 'on';
            ax.YAxis.MinorTickValues = -y_lim:obj.offset_step:y_lim;
            ax.ZMinorGrid = 'on';
            ax.ZAxis.MinorTickValues = -z_lim:obj.offset_step:z_lim;
            
            title('Offset search');
            
            view(-37.5,30);
        end
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

methods(Static, Access=private)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function rslt = check_input(particle_list_out,reference_list,tomo_list,particle_list_in)
        rslt = true;
        if( ~ischar(particle_list_out) )
            warning([particle_list_out ' must be a char array.']);
            rslt = false;
        end
        
        if( ~ischar(reference_list) )
            warning([reference_list ' must be a char array.']);
            rslt = false;
        end
        
        if( ~ischar(tomo_list) )
            warning([tomo_list ' must be a char array.']);
            rslt = false;
        end
        
        if( ~ischar(particle_list_in) )
            warning([particle_list_in ' must be a char array.']);
            rslt = false;
        end
        
        if( rslt )
        
            if( exist(reference_list,'file') )
                if( ~SUSAN.Utils.is_extension(reference_list,'refstxt') )
                    warning(['File "' reference_list '" is not a SUSAN reference list file.']);
                    rslt = false;
                end
            else
                warning(['File "' reference_list '" does not exist.']);
                    rslt = false;
            end

            if( exist(tomo_list,'file') )
                if( ~SUSAN.Utils.is_extension(tomo_list,'tomostxt') )
                    warning(['File "' tomo_list '" is not a SUSAN tomogram list file.']);
                    rslt = false;
                end
            else
                warning(['File "' tomo_list '" does not exist.']);
                    rslt = false;
            end

            if( exist(particle_list_in,'file') )
                if( ~SUSAN.Utils.is_extension(particle_list_in,'ptclsraw') )
                    warning(['File "' particle_list_in '" is not a SUSAN particle list file.']);
                    rslt = false;
                end
            else
                warning(['File "' particle_list_in '" does not exist.']);
                    rslt = false;
            end
            
        end
    end
end


end


