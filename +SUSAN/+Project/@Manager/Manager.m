classdef Manager < handle

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

properties(SetAccess=private)
    name              char    = []
    box_size          uint32  = 200
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

properties
    gpu_list          uint32  = []
    threads_per_gpu   uint32  = 1
    aligner           SUSAN.Modules.Aligner
    averager          SUSAN.Modules.Averager
    alignment_type    uint32  = 3;
    tomogram_file     char    = []
    initial_reference char    = []
    initial_particles char    = []
    cc_threshold      single  = 0.5;
    fsc_threshold     single  = 0.143;
    fsc_plot_step     single  = 5;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

properties(Access=private)
    starting_datetime
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

methods
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function obj = Manager(arg1,arg2)
        
        if( nargin == 2 )
            obj.name = arg1;
            obj.box_size = arg2;
            obj.save(arg1);
        elseif( nargin == 1 )
            obj.load(arg1);
        else
            error('1 or 2 input arguments required.');
        end
        
        obj.aligner  = SUSAN.Modules.Aligner;
        obj.averager = SUSAN.Modules.Averager;
        obj.averager.set_ctf_inversion();
        obj.averager.rec_halves = true;
        obj.averager.ssnr_s = 0.5;
        obj.averager.ssnr_f = 0;
        obj.averager.bandpass.highpass = 0;
        obj.averager.bandpass.lowpass  = obj.box_size/2-1;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function fpix_out = execute_iteration(obj,iter_number)
        
        if( nargin < 2 )
            iter_number = 0;
        end
                
        if( isempty(obj.name) )
            error('The project does not have a name');
        end
                
        if( ~exist(obj.name,'dir') )
            mkdir(obj.name);
            if( iter_number == 0 )
                iter_number = 1;
            end
        end
        
        if( iter_number == 0 )
            for iter_number = 1:999
                if( ~obj.check_iteration_exists(iter_number) )
                    break;
                end
            end
        end
        
        if( iter_number > 1 )
            if( obj.check_iteration_exists(iter_number-1) )
                [prv_refs,prv_part] = obj.get_iterations_files(iter_number-1);
            else
                error('Requested iteration %d, but no iteration %d available.',iter_number,iter_number-1);
            end
        else
            prv_refs = obj.initial_reference;
            prv_part = obj.initial_particles;
        end
        
        obj.aligner.gpu_list        = obj.gpu_list;
        obj.aligner.threads_per_gpu = obj.threads_per_gpu;
        
        obj.averager.gpu_list        = obj.gpu_list;
        obj.averager.threads_per_gpu = obj.threads_per_gpu;
        
        fprintf('\n');
        fprintf('============================\n');
        fprintf('Project %s.\n',obj.name);
        fprintf('Executing iteration %d:\n',iter_number);
        
        cur_dir = obj.get_iterations_dir(iter_number);
        if( ~exist(cur_dir,'dir') )
            mkdir(cur_dir);
        end
        
        [cur_refs,cur_part] = obj.get_iterations_files(iter_number);
        obj.aligner.working_dir = [cur_dir '/tmp/'];
        obj.averager.working_dir = [cur_dir '/tmp/'];
        
        fp_log = obj.log_init(cur_dir,iter_number);
        
        tmp_part = [cur_dir '/tmp.ptclsraw'];
        
        %%% ALIGNMENT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        obj.exec_alignment(cur_part,prv_refs,prv_part,fp_log);
        
        
        %%% PARTICLES SELECTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ptcls_count = obj.exec_selection(cur_part,tmp_part);
        
        
        %%% RECONSTRUCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        obj.exec_averaging(cur_refs,cur_dir,tmp_part,prv_refs,ptcls_count,fp_log);
        
        
        %%% FSC CALCULATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        fpix_out = obj.exec_fsc_calc(iter_number,ptcls_count,fp_log);
        
        
        %%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        fprintf('Iteration %d finished.\n',iter_number);
        obj.log_close(fp_log);
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [fsc,apix,res_fpix,res_angs] = get_fsc(obj,iter_number,ref_ix,mask)
        
        if( nargin < 4 )
            mask = [];
        end
        
        if( nargin < 3 )
            ref_ix = 1;
        end
        
        if( isempty(mask) )
            [refs_name,~] = obj.get_iterations_files(iter_number);
            refs = SUSAN.Data.ReferenceInfo.load(refs_name);
            mask = SUSAN.IO.read_mrc(refs(ref_ix).mask);
        end
        
        [h1_name,h2_name] = obj.get_name_halves(iter_number,ref_ix);
        
        [h1, ~   ] = SUSAN.IO.read_mrc(h1_name);
        [h2, apix] = SUSAN.IO.read_mrc(h2_name);
                        
        fsc = SUSAN.Utils.fsc_get(h1,h2,mask);

        fsc_ix = find(fsc<obj.fsc_threshold);
        if( ~isempty(fsc_ix) )
            res_fpix = fsc_ix(1);
            res_angs = (single(obj.box_size))*apix/(single(res_fpix));
        else
            res_fpix = size(h1,1)/2;
            res_angs = 2*apix;
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function show_fsc(obj,iter_number,ref_ix,mask)
        if( nargin < 4 )
            mask = [];
        end
        
        if( nargin < 3 )
            ref_ix = 1;
        end
        
        if( ~isscalar(iter_number) && ~isscalar(ref_ix) )
            error('ITER_NUMBER adn REF_IX cannot be vectors at the same time. Choose multiple classes or multiple iterations.')
        end
        
        hold on;
        if( ~isscalar(ref_ix) )
            for i = 1:length(ref_ix)
                [fsc,apix,~,res_angs] = obj.get_fsc(iter_number,ref_ix(i),mask);
                plot( fsc, 'LineWidth', 1.5, 'DisplayName', sprintf( ['Cl. %d [%.2f ' char(197) ']'], ref_ix(i), res_angs ) );
            end
            title( sprintf('Fourier Shell Correlation - Iteration %04d',iter_number) );
        else
            for i = 1:length(iter_number)
                [fsc,apix,~,res_angs] = obj.get_fsc(iter_number(i),ref_ix,mask);
                plot( fsc, 'LineWidth', 1.5, 'DisplayName', sprintf( ['It. %d [%.2f ' char(197) ']'], iter_number(i), res_angs ) );
            end
            title( sprintf('Fourier Shell Correlation - Class %03d',ref_ix) );
        end
        hold off;
        
        yticks(unique([0 0.143 0.5 1 obj.fsc_threshold]));
        
        t = obj.fsc_plot_step:obj.fsc_plot_step:single(obj.box_size/2);
        x_ticks_name = cell(size(t));
        for i = 1:length(t)
            x_ticks_name{i} = sprintf(['%.2f ' char(197)], single(obj.box_size)*single(apix)/single(t(i)));
        end
        
        xticks(t);
        xticklabels(x_ticks_name);
        xtickangle(45);
        
        grid on;
        
        axis([0 single(obj.box_size)/2 -0.05 1.05]);

        legend;
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function map = get_map(obj,iter_number,ref_ix)
        
        if( nargin < 3 )
            ref_ix = 1;
        end
        
        ref_name = obj.get_name_map(iter_number,ref_ix);
        map = SUSAN.IO.read_mrc(ref_name);
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function cc_read = get_cc(obj,iter_number,ref_ix)
        
        if( nargin < 3 )
            ref_ix = 1;
        end
        
        if( obj.check_iteration_exists(iter_number) )
            
            [~,part_file] = obj.get_iterations_files(iter_number);
            ptcls = SUSAN.Data.ParticlesInfo(part_file);
            
            cc_read = ptcls.ali_cc(:,:,ref_ix);
            
        else
            error('Iteration does not exist.');
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function show_proj_cc(obj,iter_number)
        
        if( obj.check_iteration_exists(iter_number) )
            
            [~,part_file] = obj.get_iterations_files(iter_number);
            ptcls = SUSAN.Data.ParticlesInfo(part_file);
            
            if( any( ptcls.prj_cc(:) > 0 ) )
                h = boxplot(permute(ptcls.prj_cc,[3 1 2]),'Symbol','r.','Widths',0.6,'OutlierSize',3,'Notch','on');
                set(h,{'linew'},{1.5});
                set(findobj(h,'LineStyle','--'),'LineStyle','-');
                grid on;
                set(gcf,'Position',[500 500 (180+20*size(ptcls.prj_cc,1)) 500]);
                title('Cross-correlation per projection');
                yticklabels('');
                xlabel('Projection');
                ylabel('Relative cross-correlation');
            end
            
        else
            error('Iteration does not exist.');
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function ptcls = get_ptcls(obj,iter_number)
        
        if( obj.check_iteration_exists(iter_number) )
            [~,part_file] = obj.get_iterations_files(iter_number);
            ptcls = SUSAN.Data.ParticlesInfo(part_file);
        else
            error('Iteration does not exist.');
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function show_cc(obj,iter_number,ref_ix,num_points)
        
        if( nargin < 4 )
            num_points = 100;
        end
        
        if( nargin < 3 )
            ref_ix = 1;
        end
        
        cc = obj.get_cc(iter_number,ref_ix);
        
        [cc_n,cc_x] = hist(cc,num_points);
        
        plot(cc_x,cc_n);
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function iter_num = get_last_iteration(obj)
        for iter_num = 1:999
            if( ~obj.check_iteration_exists(iter_num) )
                break;
            end
        end
        iter_num = iter_num - 1;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function map_file = get_name_map(obj,iter_number,ref_ix)
        if( nargin < 3 )
            ref_ix = 1;
        end
        
        if( iter_number < 1 )
            
            ref = SUSAN.Data.ReferenceInfo.load(obj.initial_reference);
            map_file = ref(ref_ix).map;
            
        elseif( obj.check_iteration_exists(iter_number) )
            
            iter_dir = obj.get_iterations_dir(iter_number);
            
            map_file = sprintf('%s/map_class%03d.mrc',iter_dir,ref_ix);
        else
            error('Iteration does not exist.');
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [h1_file,h2_file] = get_name_halves(obj,iter_number,ref_ix)
        if( nargin < 3 )
            ref_ix = 1;
        end
        
        if( iter_number < 1 )
            
            ref = SUSAN.Data.ReferenceInfo.load(obj.initial_reference);
            h1_file = ref(ref_ix).h1;
            h2_file = ref(ref_ix).h2;
            
        elseif( obj.check_iteration_exists(iter_number) )
            
            iter_dir = obj.get_iterations_dir(iter_number);
            
            h1_file = sprintf('%s/map_class%03d_half1.mrc',iter_dir,ref_ix);
            h2_file = sprintf('%s/map_class%03d_half2.mrc',iter_dir,ref_ix);
        else
            error('Iteration does not exist.');
        end
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

methods(Access=private)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function load(obj,prjname)
        
        if( exist(prjname,'dir') )
            
            fp = SUSAN.Utils.TxtRW.open_existing_file([prjname '/info.prjtxt'],'prjtxt');
            obj.name     = SUSAN.Utils.TxtRW.read_tag_char(fp,'name');
            obj.box_size = SUSAN.Utils.TxtRW.read_tag_int (fp,'box_size');
            fclose(fp);
            
            fprintf('    Accessing existing project %s [box size: %d]\n',obj.name,obj.box_size);
            
        else
            error(['Project ' prjname ' does not exist']);
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function save(obj,prjname)
        
        if( ~exist(prjname,'dir') )
            mkdir( prjname );
        else
            warning(['Project ' prjname ' already exists.']);
        end
            
        fp = SUSAN.Utils.TxtRW.create_file([prjname '/info.prjtxt'],'prjtxt');
        SUSAN.Utils.TxtRW.write_pair_char(fp,'name',     obj.name    );
        SUSAN.Utils.TxtRW.write_pair_uint(fp,'box_size', obj.box_size);
        fclose(fp);

        fprintf('    Created project %s [box size: %d]\n',obj.name,obj.box_size);
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function rslt = check_iteration_exists(obj,iter_number)
        
        iter_dir = sprintf('%s/ite_%04d',obj.name,iter_number);
        
        if( exist(iter_dir,'dir') )
            
            rslt = true;
            
            if( ~exist( [iter_dir '/reference.refstxt'], 'file' ) )
                rslt = false;
            end
            
            if( ~exist( [iter_dir '/particles.ptclsraw'], 'file' ) )
                rslt = false;
            end
            
        else
            rslt = false;
        end
        
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function iter_dir = get_iterations_dir(obj,iter_number)
        iter_dir = sprintf('%s/ite_%04d',obj.name,iter_number);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [refs,part] = get_iterations_files(obj,iter_number)
        iter_dir = obj.get_iterations_dir(iter_number);
        refs = [iter_dir '/reference.refstxt'];
        part = [iter_dir '/particles.ptclsraw'];
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function exec_alignment(obj,ptcls_out,refs,ptcls_in,fp_log)
        if( obj.alignment_type == 3 )
            fprintf('  [3D Alignment] Start:\n');
            obj.log_align3D_starts(fp_log,refs,ptcls_in);
            tic;
            obj.aligner.type = 3;
            obj.aligner.align(ptcls_out,refs,obj.tomogram_file,ptcls_in);
            t = toc;
            fprintf('  [3D Alignment] Finished using %.1f seconds (%s).\n',t,datestr(datenum(0,0,0,0,0,t),'HH:MM:SS.FFF'));
            obj.log_align_ends(fp_log,t);
        elseif( obj.alignment_type == 2 )
            fprintf('  [2D Alignment] Start:\n');
            obj.log_align2D_starts(fp_log,refs,ptcls_in);
            tic;
            obj.aligner.type = 2;
            obj.aligner.align(ptcls_out,refs,obj.tomogram_file,ptcls_in);
            t = toc;
            fprintf('  [2D Alignment] Finished using %.1f seconds (%s).\n',t,datestr(datenum(0,0,0,0,0,t),'HH:MM:SS.FFF'));
            obj.log_align_ends(fp_log,t);
        else
            fprintf(fp_log,'Invalid alignment type. Supported values: 2 and 3 (for 2D and 3D)\n');
            error('Invalid alignment type. Supported values: 2 and 3 (for 2D and 3D)');
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function ptcls_count = exec_selection(obj,ptcls_out,ptcls_tmp)
        
        fprintf('  [Aligned particles] Processing:\n');
        
        ptcls = SUSAN.Data.ParticlesInfo(ptcls_out);
        num_classes = size(ptcls.ali_cc,3);
        ptcls_count = zeros(num_classes,4);
        
        if( num_classes > 1 )
            [~,class_idx] = max( ptcls.ali_cc, [], 3 );
            ptcls.class_cix = class_idx - 1;
            ptcls.save(ptcls_out);
        end
        
        for i = 1:num_classes
            
            idx      = ptcls.class_cix == (i-1);
            half_val = ptcls.half_id(idx);
            cc_val   = ptcls.ali_cc(idx,:,i);
            
            ptcls_count(i,1) = sum(half_val(:)==1);
            ptcls_count(i,2) = sum(half_val(:)==2);
            
            th_1 = quantile( cc_val(half_val==1), 1-obj.cc_threshold );
            th_2 = quantile( cc_val(half_val==2), 1-obj.cc_threshold );
            
            half_val( half_val == 1 & cc_val<th_1 ) = 0;
            half_val( half_val == 2 & cc_val<th_2 ) = 0;
            
            ptcls.half_id(idx) = half_val;
            
            ptcls_count(i,3) = sum(half_val(:)==1);
            ptcls_count(i,4) = sum(half_val(:)==2);
            
            fprintf('    Class %d: %d particles\n',i,ptcls_count(i,1)+ptcls_count(i,2));
            fprintf('      Half 1: %d particles\n',  ptcls_count(i,1));
            fprintf('      Half 2: %d particles\n',  ptcls_count(i,2));
        end
        
        tmp = ptcls.select( ptcls.half_id > 0 );
        tmp.save(ptcls_tmp);
        fprintf('  [Aligned particles] Done.\n');
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function exec_averaging(obj,cur_refs,cur_dir,tmp_part,prv_refs,ptcls_count,fp_log)
        
        obj.log_reconstruction_starts(fp_log,size(ptcls_count,1));
        
        fprintf('  [Reconstruct Maps] Start:\n');
        tic;
        obj.averager.reconstruct([cur_dir '/map'],obj.tomogram_file,tmp_part,obj.box_size);
        t = toc;
        fprintf('  [Reconstruct Maps] Finished using %.1f seconds (%s).\n',t,datestr(datenum(0,0,0,0,0,t),'HH:MM:SS.FFF'));
        obj.log_reconstruct_time(fp_log,t);
        
        delete([cur_dir '/tmp.ptclsraw']);
            
        prv_ref_info = SUSAN.Data.ReferenceInfo.load(prv_refs);
        cur_ref_info = SUSAN.Data.ReferenceInfo.create(size(ptcls_count,1));
        
        for i = 1:size(ptcls_count,1)
            cur_ref_info(i).map  = sprintf('%s/map_class%03d.mrc',cur_dir,i);
            cur_ref_info(i).h1   = sprintf('%s/map_class%03d_half1.mrc',cur_dir,i);
            cur_ref_info(i).h2   = sprintf('%s/map_class%03d_half2.mrc',cur_dir,i);
            cur_ref_info(i).mask = prv_ref_info(i).mask;
            
            obj.log_reconstruct_ref(fp_log,i,cur_ref_info(i).h1,cur_ref_info(i).h2,cur_ref_info(i).map,ptcls_count);
            
        end
        SUSAN.Data.ReferenceInfo.save(cur_ref_info,cur_refs);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function fpix_out = exec_fsc_calc(obj,iter_number,ptcls_count,fp_log)
        
        fpix_out = zeros(size(ptcls_count,1),1);
        obj.log_fsc_starts(fp_log,size(ptcls_count,1))
        for i = 1:size(ptcls_count,1)
            [~,~,res_fpix,res_angs] = obj.get_fsc(iter_number,i);
            fprintf('  [FSC Map %d] Estimated resolution: %.3f angstroms (%d fourier pixels).\n',i,res_angs,res_fpix);
            obj.log_fsc_ref(fp_log,i,res_angs,res_fpix);
            fpix_out(i) = res_fpix;
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function fp_log = log_init(obj,cur_dir,iter_number)
        obj.starting_datetime = now();
        fp_log = fopen([cur_dir '/log.txt'],'a');
        fprintf(fp_log,'==================================\n');
        fprintf(fp_log,'[%s] Start:\n',datestr(obj.starting_datetime,'yyyy.mm.dd HH:MM:SS.FFF'));
        fprintf(fp_log,'Project %s - Iteration %d:\n',obj.name,iter_number);
        fprintf(fp_log,'- Global information:\n');
        fprintf(fp_log,'    - Box size:      %d\n',obj.box_size);
        fprintf(fp_log,'    - GPUs IDs:      %d',obj.gpu_list(1));
        for i = 2:length(obj.gpu_list)
            fprintf(fp_log,',%d',obj.gpu_list(i));
        end
        fprintf(fp_log,'\n');
        fprintf(fp_log,'    - Threads/GPU:   %d\n',obj.threads_per_gpu);
        fprintf(fp_log,'    - CC percentage: %.2f%%\n',100*obj.cc_threshold);
        fprintf(fp_log,'    - Tomogram file: %s\n',obj.tomogram_file);
        tomos = SUSAN.Data.TomosInfo(obj.tomogram_file);
        fprintf(fp_log,'        Num. tomos:  %d\n',length(tomos.tomo_id));
        fprintf(fp_log,'        Max. projs:  %d\n',size(tomos.proj_weight,1));
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function log_align3D_starts(obj,fp_log,refs_file,part_file)
        fprintf(fp_log,'- 3D Alignment Step:\n');
        fprintf(fp_log,'    - Cone Range:        %.2f\n',obj.aligner.cone.range);
        fprintf(fp_log,'    - Cone Step:         %.2f\n',obj.aligner.cone.step);
        fprintf(fp_log,'    - Inplane Range:     %.2f\n',obj.aligner.inplane.range);
        fprintf(fp_log,'    - Inplane Step:      %.2f\n',obj.aligner.inplane.step);
        fprintf(fp_log,'    - Refinement Level:  %d\n',obj.aligner.refine.level);
        fprintf(fp_log,'    - Refinement Factor: %d\n',obj.aligner.refine.factor);
        fprintf(fp_log,'    - Offset Type:       %s\n',obj.aligner.offset.type);
        fprintf(fp_log,'    - Offset Range:      [%.2f %.2f %.2f]\n',obj.aligner.offset.range(1),obj.aligner.offset.range(2),obj.aligner.offset.range(3));
        fprintf(fp_log,'    - Offset Step:       %.2f\n',obj.aligner.offset.step);
        fprintf(fp_log,'    - Bandpass range:    [%.1f %1f]\n',obj.aligner.bandpass.highpass,obj.aligner.bandpass.lowpass);
        obj.log_refs_file(fp_log,refs_file);
        obj.log_part_file(fp_log,part_file);
        fprintf(fp_log,'    Starting at %s\n',datestr(now(),'yyyy.mm.dd HH:MM:SS.FFF'));
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function log_align2D_starts(obj,fp_log,refs_file,part_file)
        fprintf(fp_log,'- 2D Alignment Step:\n');
        fprintf(fp_log,'    - Cone Range:        %.2f\n',obj.aligner.cone.range);
        fprintf(fp_log,'    - Cone Step:         %.2f\n',obj.aligner.cone.step);
        fprintf(fp_log,'    - Inplane Range:     %.2f\n',obj.aligner.inplane.range);
        fprintf(fp_log,'    - Inplane Step:      %.2f\n',obj.aligner.inplane.step);
        fprintf(fp_log,'    - Offset Radius:     %.2f\n',obj.aligner.offset.range(1));
        fprintf(fp_log,'    - Bandpass range:    [%.1f %.1f]\n',obj.aligner.bandpass.highpass,obj.aligner.bandpass.lowpass);
        obj.log_refs_file(fp_log,refs_file);
        obj.log_part_file(fp_log,part_file);
        fprintf(fp_log,'    Starting at %s\n',datestr(now(),'yyyy.mm.dd HH:MM:SS.FFF'));
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function log_align_ends(~,fp_log,exec_time)
        fprintf(fp_log,'    Finished at %s.\n',datestr(now(),'yyyy.mm.dd HH:MM:SS.FFF'));
        fprintf(fp_log,'    Execution time: %.1f seconds (%s).\n',exec_time,datestr(datenum(0,0,0,0,0,exec_time),'HH:MM:SS.FFF'));
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function log_reconstruction_starts(~,fp_log,num_classes)
        if( num_classes > 1 )
            fprintf(fp_log,'- 3D Reconstruction Step (%d references):\n',num_classes);
        else
            fprintf(fp_log,'- 3D Reconstruction Step (1 reference):\n');
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function log_reconstruct_ref(~,fp_log,ref_ix,h1_map,h2_map,final_map,ptcls_count)
        fprintf(fp_log,'    - Reference %2d\n',ref_ix);
        fprintf(fp_log,'        Half 1:    %s [%d particles]\n'  ,h1_map   ,ptcls_count(ref_ix,3));
        fprintf(fp_log,'        Half 2:    %s [%d particles]\n'  ,h2_map   ,ptcls_count(ref_ix,4));
        fprintf(fp_log,'        Final Map: %s   [%d particles]\n',final_map,ptcls_count(ref_ix,3)+ptcls_count(ref_ix,4));
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function log_reconstruct_time(~,fp_log,exec_time)
        fprintf(fp_log,'        Execution time: %.1f seconds (%s).\n',exec_time,datestr(datenum(0,0,0,0,0,exec_time),'HH:MM:SS.FFF'));
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function log_fsc_starts(~,fp_log,num_classes)
        if( num_classes > 1 )
            fprintf(fp_log,'- FSC Calculation (%d references):\n',num_classes);
        else
            fprintf(fp_log,'- FSC Calculation (1 reference):\n');
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function log_fsc_ref(~,fp_log,ref_ix,res_angs,res_fpix)
        fprintf(fp_log,'    - Reference %2d: %7.3f angstroms [%d fourier pixels]\n',ref_ix,res_angs,res_fpix);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function log_refs_file(~,fp_log,refs_file)
        ref = SUSAN.Data.ReferenceInfo.load(refs_file);
        fprintf(fp_log,'    - Reference file:    %s\n',refs_file);
        fprintf(fp_log,'    - Num. References:   %d\n',length(ref));
        for i = 1:length(ref)
            fprintf(fp_log,'        Ref. %2d:         %s\n',i,ref(i).map);
            fprintf(fp_log,'        Mask %2d:         %s\n',i,ref(i).mask);
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function log_part_file(~,fp_log,part_file)
        ptcls = SUSAN.Data.ParticlesInfo(part_file);
        fprintf(fp_log,'    - Particles file:    %s\n',part_file);
        fprintf(fp_log,'    - Num. Particles:    %d\n',length(ptcls.ptcl_id));
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function log_close(obj,fp_log)
        ending_time = now();
        fprintf(fp_log,'[%s] Finished. Total time: %s\n',datestr(ending_time,'yyyy.mm.dd HH:MM:SS.FFF'),datestr(ending_time-obj.starting_datetime,'HH:MM:SS.FFF'));
        fclose(fp_log);
    end
end

    
end


