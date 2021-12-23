classdef CtfRefiner < handle
% SUSAN.Modules.CtfRefiner Performs per-particle-per-projection CTF refinement.
%    Performs per-particle-per-projection CTF refinement.
%
% SUSAN.Modules.CtfRefiner Properties:
%    gpu_list        - (vector) list of GPU IDs to use for alignment.
%    bandpass        - (struct) frequency range to use.
%    halfsets        - (bool)   refine using independent halfsets.
%    threads_per_gpu - (scalar) multiple threads per GPU.
%    ranges          - (struct) ranges for for defocus search.
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
    ranges                  = struct('defocus',1000,'angles',20);
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
        
        cmd = [cmd ' -def_range '  sprintf('%f',obj.ranges.defocus)];
        cmd = [cmd ' -ang_range '  sprintf('%f',obj.ranges.angles)];
        
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

end


