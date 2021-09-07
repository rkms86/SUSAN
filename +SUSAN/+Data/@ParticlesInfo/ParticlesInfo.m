 classdef ParticlesInfo < handle
% SUSAN.Data.ParticlesInfo Holds the information of all the particles.
%    Holds the information of all the particles used in SUSAN.
%
%    For this documentation:
%        P: number of particles.
%        R: number of references/classes.
%        K: maximum number of projections.
%
% SUSAN.Data.ParticlesInfo Properties:
%    ptcl_id   - (Px1 Array)    Particle ID.
%    tomo_id   - (Px1 Array)    Tomogram ID where the particles is located.
%    tomo_cix  - (Px1 Array)    C-style index of the Tomogram in the TomogramsInfo object.
%    position  - (Px3 Matrix)   Position of the particle , relative to the center (Angstroms).
%    class_cix - (Px1 Array)    C-style index of the particle class/reference.
%    half_id   - (Px1 Array)    Particle half map (1 or 2).
%    extra_1   - (Px1 Array)    Extra annotation 1.
%    extra_2   - (Px1 Array)    Extra annotation 2.
%    ali_eZYZ  - (Px3xR Matrix) 3D Rotation for each reference, in eulerZYZ.
%    ali_t     - (Px3xR Matrix) 3D Shift for each reference (Angstroms).
%    ali_cc    - (Px1xR Matrix) Cross correlation value for each reference.
%    ali_w     - (Px1xR Matrix) Weight of the particle for each reference.
%    prj_eZYZ  - (Kx3xP Matrix) 3D Rotation of each projection in eulerZYZ.
%    prj_t     - (Kx2xP Matrix) 2D Shift of each projection (Angstroms).
%    prj_cc    - (Kx1xP Matrix) Cross correlaiton value for each projection.
%    prj_w     - (Kx1xP Matrix) Weights to be applied to each projection.
%    defocus   - (Kx8xP Matrix) Defocus information for each projection.
%    Notes:
%    - The defocus matrix is composed by 8 columns, each holding the
%      following information:
%         Column 1: Defocus U.
%         Column 2: Defocus V.
%         Column 3: Defocus angle.
%         Column 4: Defocus phase shift.
%         Column 5: Defocus BFactor.
%         Column 6: Exposure Filter.
%         Column 7: Max. Resolution (angstroms).
%         Column 8: Fitting Score.
%
% SUSAN.Data.ParticlesInfo Methods:
%    ParticlesInfo     - (Constructor) creates a ParticlesInfo.
%    save              - Saves the current ParticlesInfo in a file.
%    n_ptcls           - Returns the number of particles.
%    n_refs            - Returns the number of references.
%    max_projs         - Returns the maximum number of projections.
%    get_copy          - Creates a deep-copy of the current ParticlesInfo object.
%    select            - Creates a subset of the current ParticlesInfo object.
%    set_weights       - Sets the weight for each particle.
%    halfsets_by_Y     - Sets the halfsets according to the particles' Y position.
%    halfsets_even_odd - Sets the halfsets in an even/odd fashion.
%    update_position   - Update the particles' position with the shifts (ali_t). 
%    update_defocus    - Update defocus according to the particles' Z position. 
%    append_ptcls      - Append an existing particles obj to the current one.
%    export_dyntbl     - Exports to a dynamo table.
%
% SUSAN.Data.ParticlesInfo Static Methods:
%    grid2D               - Creates a grid at the Z=0 plane of each tomogram.
%    grid3D               - Creates a grid distributed evenly on each tomogram.

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

properties
    ptcl_id   uint32
    tomo_id   uint32
    tomo_cix  uint32
    position  single
    class_cix uint32
    half_id   uint32
    extra_1   single
    extra_2   single
    ali_eZYZ  single
    ali_t     single
    ali_cc    single
    ali_w     single
    prj_eZYZ  single
    prj_t     single
    prj_cc    single
    prj_w     single
    defocus   single
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function obj = ParticlesInfo(arg1,arg2,arg3)
    % ParticlesInfo [Constructor] Creates ParticlesInfo object:
    %   ParticlesInfo(NUM_PTCLS,MAX_PROJS,NUM_REFS) Creates a ParticlesInfo
    %   object of NUM_PTCLS particles, with maximum MAX_PROJS projections, 
    %   and NUM_REFS references (classes). The object will be empty.
    %   ParticlesInfo(DYN_TBL,TOMOS_INFO) Imports the information of a
    %   dynamo_table into a ParticlesInfo object, using the information of
    %   TOMOS_INFO, a SUSAN.Data.TomosInfo object.
    %   for NUM_TOMOS tomograms, with a maximum of MAX_PROJS projections.
    %   ParticlesInfo(FILENAME) Loads a ParticlesInfo object from FILENAME.
    %   File extension: 'ptclsraw'.
    %
    %   See also SUSAN.Data.TomosInfo.
        
        if( nargin == 3 )
            obj.allocate(arg1,arg2,arg3);
        elseif( nargin == 2 )
            obj.import_dynamo_table(arg1,arg2);
        elseif( nargin == 1 )
            if( ischar(arg1) )
                obj.internal_load(arg1);
            end
        else
            error('Wrong number of inputs.');
        end
        
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function save(obj,filename)     
    % SAVE saves the current ParticlesInfo object into a file.
    %   SAVE(FILENAME) Saves the ParticlesInfo object in FILENAME. File
    %   extension: 'ptclsraw'.
    %
    %   See also SUSAN.Data.ParticlesInfo, SUSAN.Data.ParticlesInfo.load.
    
        working_filename = SUSAN.Utils.force_extension(filename,'ptclsraw');
        ParticlesInfo_save_by_blocks(working_filename, ...
            [obj.ptcl_id, obj.tomo_id, obj.tomo_cix, obj.class_cix, obj.half_id],...
            cat(2,obj.position, obj.extra_1, obj.extra_2), ...
            cat(2,obj.ali_eZYZ,obj.ali_t,obj.ali_cc,obj.ali_w), ...
            cat(2,obj.prj_eZYZ,obj.prj_t,obj.prj_cc,obj.prj_w), ...
            obj.defocus );
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function n = n_ptcls(obj)     
    % N_PTCLS returns the number of particles.
    %   N = N_PTCLS() return the number of particles.
        
        n = length(obj.ptcl_id);
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function n = n_refs(obj)     
    % N_REFS returns the number of references.
    %   N = N_REFS() return the number of references.
    %
    %   See also SUSAN.Data.ParticlesInfo, SUSAN.Data.ParticlesInfo.load.
        
        n = size(obj.ali_eZYZ,3);
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function n = max_projs(obj)     
    % MAX_PROJS returns the maximum number of projections.
    %   N = N_PROJS() returns the maximum number of projections.
    %
    %   See also SUSAN.Data.ParticlesInfo, SUSAN.Data.ParticlesInfo.load.
        
        n = size(obj.prj_eZYZ,1);
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function ptcls_obj = get_copy(obj)     
    % SELECT returns a new ParticlesInfo object, a copy of the current one.
    %   PTCLS_OBJ = GET_COPY(IX) returns a new ParticlesInfo object, a
    %   deep-copy of the current.
    %
    %   See also SUSAN.Data.ParticlesInfo.
        
        if( ~isvector(ix) )
            error('Error in the IX argument: it must be a vector');
        end
    
        ptcls_obj = SUSAN.Data.ParticlesInfo(obj.n_ptcls(),obj.max_projs(),obj.n_refs());
        
        ptcls_obj.ptcl_id   = obj.ptcl_id;
        ptcls_obj.tomo_id   = obj.tomo_id;
        ptcls_obj.tomo_cix  = obj.tomo_cix;
        ptcls_obj.position  = obj.position;
        ptcls_obj.class_cix = obj.class_cix;
        ptcls_obj.half_id   = obj.half_id;
        ptcls_obj.extra_1   = obj.extra_1;
        ptcls_obj.extra_2   = obj.extra_2;
        ptcls_obj.ali_eZYZ  = obj.ali_eZYZ;
        ptcls_obj.ali_t     = obj.ali_t;
        ptcls_obj.ali_cc    = obj.ali_cc;
        ptcls_obj.ali_w     = obj.ali_w;
        ptcls_obj.prj_eZYZ  = obj.prj_eZYZ;
        ptcls_obj.prj_t     = obj.prj_t;
        ptcls_obj.prj_cc    = obj.prj_cc;
        ptcls_obj.prj_w     = obj.prj_w;
        ptcls_obj.defocus   = obj.defocus;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function ptcls_obj = select(obj,ix)     
    % SELECT returns a new ParticlesInfo object, a subset of the current one.
    %   PTCLS_OBJ = SELECT(IX) returns a new ParticlesInfo object, a subset
    %   of the current one according to the column vector IX.
    %
    %   See also SUSAN.Data.ParticlesInfo.
        
        if( ~isvector(ix) )
            error('Error in the IX argument: it must be a vector');
        end
    
        new_ptcl_id = obj.ptcl_id(ix(:));
        ptcls_obj = SUSAN.Data.ParticlesInfo(length(new_ptcl_id),obj.max_projs(),obj.n_refs());
        
        ptcls_obj.ptcl_id(:)      = obj.ptcl_id   (ix);
        ptcls_obj.tomo_id(:)      = obj.tomo_id   (ix);
        ptcls_obj.tomo_cix(:)     = obj.tomo_cix  (ix);
        ptcls_obj.position(:,:)   = obj.position  (ix,:);
        ptcls_obj.class_cix(:)    = obj.class_cix (ix);
        ptcls_obj.half_id(:)      = obj.half_id   (ix);
        ptcls_obj.extra_1(:)      = obj.extra_1   (ix);
        ptcls_obj.extra_2(:)      = obj.extra_2   (ix);
        ptcls_obj.ali_eZYZ(:,:)   = obj.ali_eZYZ  (ix,:);
        ptcls_obj.ali_t(:,:)      = obj.ali_t     (ix,:);
        ptcls_obj.ali_cc(:)       = obj.ali_cc    (ix,:);
        ptcls_obj.ali_w(:)        = obj.ali_w     (ix,:);
        ptcls_obj.prj_eZYZ(:,:,:) = obj.prj_eZYZ  (:,:,ix);
        ptcls_obj.prj_t(:,:,:)    = obj.prj_t     (:,:,ix);
        ptcls_obj.prj_cc(:,:,:)   = obj.prj_cc    (:,:,ix);
        ptcls_obj.prj_w(:,:,:)    = obj.prj_w     (:,:,ix);
        ptcls_obj.defocus(:,:,:)  = obj.defocus   (:,:,ix);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_weights(obj,in_wgt)     
    % SET_WEIGHTS sets the wwights for all the particles.
    %   SELECT(IN_WGT) set each element in vector IN_WGT as the weight of
    %   each projection for the particle associated to that element.
    %
    %   See also SUSAN.Data.ParticlesInfo.
        
        if( ~isvector(in_wgt) )
            error('Error in the in_wgt argument: it must be a vector');
        end
        
        if( length(in_wgt) ~= obj.n_ptcls )
            error('Error in the in_wgt argument: its length must be the asme as the number of particles');
        end
        
        for i = 1:obj.max_projs()
            ix = obj.prj_w(i,:,:)>0;
            obj.prj_w(i,:,ix) = in_wgt(ix);
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function halfsets_by_Y(obj)     
    % HALFSETS_BY_Y sets the halves according the particles position on the
    % Y axis.
        
        tomos_id = unique( obj.tomo_id );
        
        h_id = [];
        
        for i = 1:length(tomos_id)
            t_idx = (obj.tomo_id == tomos_id(i));
            cur_y = obj.position(t_idx,2);
            cur_hid = ones(sum(t_idx),1);
            cur_hid( cur_y > quantile(cur_y,0.5) ) = 2;
            h_id = [h_id; cur_hid];
        end
        
        obj.half_id = h_id;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function halfsets_even_odd(obj)     
    % HALFSETS_EVEN_ODD sets the halves in an even/odd fashion.
        
        obj.half_id(1:2:end) = 1;
        obj.half_id(2:2:end) = 2;
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function append_ptcls(obj,ptcls)
    % APPEND_PTCLS Appends an existing particles obj to the current one.
    %   APPEND_PTCLS(PTCLS) Adds the particles in PTCLS at the end of the
    %   current one.

        if( ~isa(ptcls,'SUSAN.Data.ParticlesInfo') )
            error('Input must be a ParticlesInfo object');
        end
    
        if( obj.n_refs ~= ptcls.n_refs )
            error('Inconsistency: different number of classes');
        end
        
        obj.ptcl_id   = cat(1, obj.ptcl_id  , ptcls.ptcl_id  );
        obj.tomo_id   = cat(1, obj.tomo_id  , ptcls.tomo_id  );
        obj.tomo_cix  = cat(1, obj.tomo_cix , ptcls.tomo_cix );
        obj.position  = cat(1, obj.position , ptcls.position );
        obj.class_cix = cat(1, obj.class_cix, ptcls.class_cix);
        obj.half_id   = cat(1, obj.half_id  , ptcls.half_id  );
        obj.extra_1   = cat(1, obj.extra_1  , ptcls.extra_1  );
        obj.extra_2   = cat(1, obj.extra_2  , ptcls.extra_2  );
        obj.ali_eZYZ  = cat(1, obj.ali_eZYZ , ptcls.ali_eZYZ );
        obj.ali_t     = cat(1, obj.ali_t    , ptcls.ali_t    );
        obj.ali_cc    = cat(1, obj.ali_cc   , ptcls.ali_cc   );
        obj.ali_w     = cat(1, obj.ali_w    , ptcls.ali_w    );
        obj.prj_eZYZ  = cat(3, obj.prj_eZYZ , ptcls.prj_eZYZ );
        obj.prj_t     = cat(3, obj.prj_t    , ptcls.prj_t    );
        obj.prj_cc    = cat(3, obj.prj_cc   , ptcls.prj_cc   );
        obj.prj_w     = cat(3, obj.prj_w    , ptcls.prj_w    );
        obj.defocus   = cat(3, obj.defocus  , ptcls.defocus  );
        
        [~,ix] = sortrows([obj.tomo_id obj.ptcl_id]);
        
        
        obj.ptcl_id   = obj.ptcl_id(ix);
        obj.tomo_id   = obj.tomo_id(ix);
        obj.tomo_cix  = obj.tomo_cix(ix);
        obj.position  = obj.position(ix,:);
        obj.class_cix = obj.class_cix(ix);
        obj.half_id   = obj.half_id(ix);
        obj.extra_1   = obj.extra_1(ix);
        obj.extra_2   = obj.extra_2(ix);
        obj.ali_eZYZ  = obj.ali_eZYZ(ix,:,:);
        obj.ali_t     = obj.ali_t(ix,:,:);
        obj.ali_cc    = obj.ali_cc(ix,:,:);
        obj.ali_w     = obj.ali_w(ix,:,:);
        obj.prj_eZYZ  = obj.prj_eZYZ(:,:,ix);
        obj.prj_t     = obj.prj_t(:,:,ix);
        obj.prj_cc    = obj.prj_cc(:,:,ix);
        obj.prj_w     = obj.prj_w(:,:,ix);
        obj.defocus   = obj.defocus(:,:,ix);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function update_position(obj)
    % UPDATE_POSITION updates the particles' position with the estimated shifts.
    %   UPDATE_POSITION() Updates the position with the current shift
    %   values of the particles. This operation is only valid when one
    %   reference is available. Pseudocode:
    %       particles.position <- particles.position + particles.ali_t
    %       particles.ali_t <- 0
    %
    %   See also SUSAN.Data.ParticlesInfo.
    
        if( obj.n_refs() > 1 )
            
            error(['Particles has ' num2str(obj.n_refs()) ' classes. It must have only 1.' ]);
        end
        
        obj.position = obj.position + obj.ali_t;
        obj.ali_t(:) = 0;
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function update_defocus(obj,tomos_list)
    % UPDATE_DEFOCUS updates the defocus information of each particle.
    %   UPDATE_DEFOCUS(TOMOS_LIST) updates the defocus information of the
    %   particles from TOMOS_LIST and enables all projections.
    %
    %   See also SUSAN.Data.ParticlesInfo.
    
        if( ~isa(tomos_list,'SUSAN.Data.TomosInfo') )
            error('Second argument must be a SUSAN.Data.TomosInfo object.');
        end
        
        tomo_ix     = obj.tomo_cix + 1;
        obj.prj_w   = tomos_list.proj_weight(:,:,tomo_ix);
        cur_pos     = obj.position+obj.ali_t(:,:,1);
        obj.defocus = ParticlesInfo_defocus_per_ptcl(obj.tomo_cix,cur_pos,tomos_list.proj_eZYZ,tomos_list.defocus,single(-1));
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function tbl = export_dyntbl(obj,tomos_list,ref_idx)
    % EXPORT_DYNTBL Create a dynamo table from the curent ParticlesInfo.
    %   TBL = EXPORT_DYNTBL(TOMOS_LIST,REF_IDX) Export the current
    %   alignment information for class REF_IDX (default = 1) into a dynamo
    %   table. TOMOS_LIST is a TomosInfo object.
    %   NOTE: It needs uses the dynamo_table_blank function.
    %
    %   See also SUSAN.Data.ParticlesInfo.
    
        if( nargin < 3 )
            ref_idx = 1;
        end
    
        if( ~isa(tomos_list,'SUSAN.Data.TomosInfo') )
            error('Second argument must be a SUSAN.Data.TomosInfo object.');
        end
        
        tbl = dynamo_table_blank( obj.n_ptcls );
        
        pos = bsxfun(@times, obj.position, 1./tomos_list.pix_size(obj.tomo_cix+1));
        shf = bsxfun(@times, obj.ali_t(:,:,ref_idx), 1./tomos_list.pix_size(obj.tomo_cix+1));
        
        tbl(:,20) = obj.tomo_id;
        tbl(:,23) = obj.half_id;
        tbl(:,[24 25 26]) = pos + single( tomos_list.tomo_size(obj.tomo_cix+1,:) )/2;
        tbl(:,[4 5 6]) = shf;
        
        tbl(:,10) = obj.ali_cc(:,:,ref_idx);
        tbl(:,[7 8 9]) = ParticlesInfo_euZYZ_2_euDYN(obj.ali_eZYZ(:,:,ref_idx));       
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function rslt = x(obj,ref_idx)    
    % X returns the X position os all the particles.
    %   RSLT = X(REF_IDX) returns position(:,1)+ali_t(:,1). REF_IDX defines
    %   which ALI_T will be used (for multi-reference projects). By default
    %   REF_IDX = 1.
        
        if( nargin < 2 )
            ref_idx = 1;
        end
        
        rslt = obj.position(:,1) + obj.ali_t(:,1,ref_idx);
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function rslt = y(obj,ref_idx)     
    % Y returns the Y position os all the particles.
    %   RSLT = X(REF_IDX) returns position(:,2)+ali_t(:,2). REF_IDX defines
    %   which ALI_T will be used (for multi-reference projects). By default
    %   REF_IDX = 1.
        
        if( nargin < 2 )
            ref_idx = 1;
        end
        
        rslt = obj.position(:,2) + obj.ali_t(:,2,ref_idx);
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function rslt = z(obj,ref_idx)     
    % Z returns the Z position os all the particles.
    %   RSLT = Z(REF_IDX) returns position(:,3)+ali_t(:,3). REF_IDX defines
    %   which ALI_T will be used (for multi-reference projects). By default
    %   REF_IDX = 1.
        
        if( nargin < 2 )
            ref_idx = 1;
        end
        
        rslt = obj.position(:,3) + obj.ali_t(:,3,ref_idx);
        
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

methods(Static)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function rslt = grid2D(step,tomos_list)
    % GRID2D Creates a grid at the Z=0 plane of each tomogram.
    %   PTCLS = GRID2D(STEP,TOMOS_LIST)
    %
    %   See also SUSAN.Data.ParticlesInfo.
    
        if( ~isa(tomos_list,'SUSAN.Data.TomosInfo') )
            error('Second argument must be a SUSAN.Data.TomosInfo object.');
        end

        pix_size = min(tomos_list.pix_size);
        range = max( single(tomos_list.tomo_size(:,1:2)), [], 1 )/2;
        r_x = step*floor((range(1) - ceil(step/2))/step);
        r_y = step*floor((range(2) - ceil(step/2))/step);
        [X,Y] = meshgrid(-r_x:step:r_x,-r_y:step:r_y);
        X = X(:)*pix_size;
        Y = Y(:)*pix_size;
        
        num_pts = numel(X);
        num_tom = length(tomos_list.tomo_id);
        num_prj = size(tomos_list.proj_weight,1);
        
        rslt = SUSAN.Data.ParticlesInfo(num_pts*num_tom,num_prj,1);
        
        tomo_id_list = unique(tomos_list.tomo_id)';
        rslt.tomo_id = reshape(repmat(tomo_id_list,[num_pts 1]),[num_pts*length(tomo_id_list) 1]);
        rslt.tomo_cix = reshape(repmat(0:(length(tomo_id_list)-1),[num_pts 1]),[num_pts*length(tomo_id_list) 1]);
        
        rslt.position(:,1:2) = repmat([X Y],[num_tom 1]);
        rslt.ali_cc(:) = 1e-6;
        
        rslt.update_defocus(tomos_list);
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function rslt = grid3D(step,tomos_list)
    % GRID3D Creates a grid distributed evenly on each tomogram.
    %   PTCLS = GRID3D(STEP,TOMOS_LIST)
    %
    %   See also SUSAN.Data.ParticlesInfo.
    
        if( ~isa(tomos_list,'SUSAN.Data.TomosInfo') )
            error('Second argument must be a SUSAN.Data.TomosInfo object.');
        end

        pix_size = min(tomos_list.pix_size);
        range = max( single(tomos_list.tomo_size), [], 1 )/2;
        r_x = step*floor((range(1) - ceil(step/2))/step);
        r_y = step*floor((range(2) - ceil(step/2))/step);
        r_z = step*floor((range(3) - ceil(step/2))/step);
        [X,Y,Z] = meshgrid(-r_x:step:r_x,-r_y:step:r_y,-r_z:step:r_z);
        X = X(:)*pix_size;
        Y = Y(:)*pix_size;
        Z = Z(:)*pix_size;
        
        num_pts = numel(X);
        num_tom = length(tomos_list.tomo_id);
        num_prj = size(tomos_list.proj_weight,1);
        
        rslt = SUSAN.Data.ParticlesInfo(num_pts*num_tom,num_prj,1);
        
        tomo_id_list = unique(tomos_list.tomo_id)';
        rslt.tomo_id = reshape(repmat(tomo_id_list,[num_pts 1]),[num_pts*length(tomo_id_list) 1]);
        rslt.tomo_cix = reshape(repmat(0:(length(tomo_id_list)-1),[num_pts 1]),[num_pts*length(tomo_id_list) 1]);
        
        rslt.position = repmat([X Y Z],[num_tom 1]);
        rslt.ali_cc(:) = 1e-6;
        
        rslt.update_defocus(tomos_list);
        
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

methods(Access=private)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function allocate(obj,num_ptcls,max_proj,num_refs)
        obj.ptcl_id   = (1:num_ptcls)';
        obj.tomo_id   = ones (num_ptcls,1);
        obj.tomo_cix  = zeros(num_ptcls,1);
        obj.position  = zeros(num_ptcls,3);
        obj.class_cix = zeros(num_ptcls,1);
        obj.half_id   = 2-mod(obj.ptcl_id,2);
        obj.extra_1   = zeros(num_ptcls,1);
        obj.extra_2   = zeros(num_ptcls,1);
        obj.ali_eZYZ  = zeros(num_ptcls,3,num_refs);
        obj.ali_t     = zeros(num_ptcls,3,num_refs);
        obj.ali_cc    = zeros(num_ptcls,1,num_refs);
        obj.ali_w     = ones(num_ptcls,1,num_refs);
        obj.prj_eZYZ  = zeros(max_proj,3,num_ptcls);
        obj.prj_t     = zeros(max_proj,2,num_ptcls);
        obj.prj_cc    = zeros(max_proj,1,num_ptcls);
        obj.prj_w     = zeros(max_proj,1,num_ptcls);
        obj.defocus   = zeros(max_proj,8,num_ptcls);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function import_dynamo_table(obj,dyn_tbl,tomos_list)
        
        if( ~isa(tomos_list,'SUSAN.Data.TomosInfo') )
            error('Second argument must be a SUSAN.Data.TomosInfo object.');
        end
        
        if( ~isa(dyn_tbl,'double') )
            error('First argument must be a dynamo_table (double matrix).');
        end
        
        dyn_tbl = sortrows(dyn_tbl,[20 1]);
        
        obj.allocate(size(dyn_tbl,1),size(tomos_list.proj_eZYZ,1),1);
        
        obj.ptcl_id = dyn_tbl(:,1);
        
        obj.tomo_id = uint32(dyn_tbl(:,20));
        
        tomo_id_list = unique(obj.tomo_id);
        tomo_id_LUT  = zeros(max(tomo_id_list),1);
        for i = 1:length(tomo_id_list)
            tomo_id_LUT( tomo_id_list(i) ) = i;
        end
        tomo_ix = tomo_id_LUT(obj.tomo_id);
        obj.tomo_cix = tomo_ix-1;        
        
        if( length(tomo_id_list) > max(obj.tomo_cix+1) )
            error('Inconsistencies between the TOMO_ID on the table and the tomoinfo');
        end
        
        obj.position = single( dyn_tbl(:,[24 25 26]) + dyn_tbl(:,[4 5 6]) ) - single( tomos_list.tomo_size(tomo_ix,:) )/2;
        obj.position = bsxfun(@times, obj.position, tomos_list.pix_size(tomo_ix));
        
        obj.ali_eZYZ = ParticlesInfo_euDYN_2_euZYZ(single(dyn_tbl(:,[7 8 9])));
        obj.ali_cc   = dyn_tbl(:,10);
        
        obj.prj_w    = tomos_list.proj_weight(:,:,tomo_ix);
        
        obj.defocus = ParticlesInfo_defocus_per_ptcl(obj.tomo_cix,obj.position,tomos_list.proj_eZYZ,tomos_list.defocus,single(-1));
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function internal_load(obj,filename)
        
        working_filename = SUSAN.Utils.force_extension(filename,'ptclsraw');
        [u23_blk, flt_blk, ali_blk, prj_blk, ctf_blk] = ParticlesInfo_load_by_blocks(working_filename);
            
        obj.ptcl_id   = u23_blk(:,1);
        obj.tomo_id   = u23_blk(:,2);
        obj.tomo_cix  = u23_blk(:,3);
        obj.position  = flt_blk(:,[1 2 3]);
        obj.class_cix = u23_blk(:,4);
        obj.half_id   = u23_blk(:,5);
        obj.extra_1   = flt_blk(:,4);
        obj.extra_2   = flt_blk(:,5);
        obj.ali_eZYZ  = ali_blk(:,[1 2 3],:);
        obj.ali_t     = ali_blk(:,[4 5 6],:);
        obj.ali_cc    = ali_blk(:,7,:);
        obj.ali_w     = ali_blk(:,8,:);
        obj.prj_eZYZ  = prj_blk(:,[1 2 3],:);
        obj.prj_t     = prj_blk(:,[4 5],:);
        obj.prj_cc    = prj_blk(:,6,:);
        obj.prj_w     = prj_blk(:,7,:);
        obj.defocus   = ctf_blk;
    end
    
end

 end





 
 
