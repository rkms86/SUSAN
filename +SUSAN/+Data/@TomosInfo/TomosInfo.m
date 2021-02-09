classdef TomosInfo < handle
% SUSAN.Data.TomosInfo   Holds the information of the tomograms.
%    Holds the information of the tomograms/stacks to be used in SUSAN.
%
%    For this documentation:
%        T: number of tomograms/stacks.
%        K: maximum number of projections in the stacks.
%
% SUSAN.Data.TomosInfo Properties:
%    tomo_id      - (Tx1 Array)    Tomograms' ID.
%    stack_file   - (Tx1 Cell)     Stacks' filename (.st, .ali or .mrc).
%    stack_size   - (Tx3 Matrix)   Stacks' size (pixels).
%    num_proj     - (Tx1 Array)    Number of projection of each stack.
%    tomo_size    - (Tx3 Matrix)   Tomograms' size (voxels).
%    pix_size     - (Tx1 Array)    Pixel size in angstroms.
%    proj_eZYZ    - (Kx3xT Matrix) 3D Rotation of each projection in eulerZYZ.
%    proj_shift   - (Kx2xT Matrix) 2D Shift of each projection.
%    proj_weight  - (Kx1xT Matrix) Weights to be applied to each projection.
%    voltage      - (Tx1 Array)    Voltage in KV.
%    sph_aberr    - (Tx1 Array)    Spherical aberration.
%    amp_contrast - (Tx1 Array)    Amplitude contrast.
%    defocus      - (Kx8xT Matrix) Defocus information for each projection.
%    Notes:
%    - tomo_id is related to the tomogram number entry in the dynamo_table.
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
% SUSAN.Data.TomosInfo Methods:
%    TomosInfo   - (Constructor) creates a TomoInfo from a file or from T and K.
%    set_stack   - Sets a stack file for a tomogram entry.
%    set_angles  - Loads and sets the rotations for a tomogram entry.
%    set_defocus - Loads and sets the defocus information for a tomogram entry.
%    save        - Saves the TomosInfo into a file.
%
% See also dthelp
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

properties
    tomo_id      uint32
    stack_file
    stack_size   uint32
    num_proj     uint32
    tomo_size    uint32
    pix_size     single
    proj_eZYZ    single
    proj_shift   single
    proj_weight  single
    voltage      single
    sph_aberr    single
    amp_contrast single
    defocus      single
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function obj = TomosInfo(arg1,arg2)
    % TomosInfo [Constructor] Creates TomosInfo object:
    %   TomosInfo(NUM_TOMOS,MAX_PROJS) Allocates an empty TomosInfo object
    %   for NUM_TOMOS tomograms, with a maximum of MAX_PROJS projections.
    %   TomosInfo(FILENAME) Loads a TomosInfo object from FILENAME. File
    %   extension: 'tomostxt'.
    %
    %   See also SUSAN.Data.TomosInfo, SUSAN.Data.TomosInfo.load.
        
        if( nargin == 2 )
            obj.allocate_properties(arg1,arg2);
        elseif( nargin == 1 )
            if( ischar(arg1) )
                obj.internal_load(arg1);
            end
        else
            error('Wrong number of inputs.');
        end
        
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_stack(obj,tomo_ix,filename)
    % SET_STACK Sets the stack file for a tomogram.
    %   SET_STACK(TOMO_IX,FILENAME) Sets the stack_file to FILENAME for the
    %   TOMO_IX-th entry. It reads FILENAME and sets up stack_size, num_proj
    %   and pix_size too.
    %
    %   See also SUSAN.Data.TomosInfo.set_angles, SUSAN.Data.TomosInfo.set_defocus.
        if( SUSAN.Utils.exist_file(filename) )
            [dims, ~, apix] = SUSAN.Utils.get_mrc_info(filename);
            obj.stack_file{tomo_ix}   = filename;
            obj.stack_size(tomo_ix,:) = dims;
            obj.num_proj(tomo_ix)     = dims(3);
            obj.pix_size(tomo_ix)     = apix;
        else
            error('File %s does not exist.',filename);
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_angles(obj,tomo_ix,tltname,xfname,xfapix)
    % SET_ANGLES Sets the rotation angles for a tomogram.
    %   SET_ANGLES(TOMO_IX,TLTFILE) Reads a TLTFILE and sets up the
    %     TOMO_IX-th entry of proj_eZYZ. TLTFILE must be an IMOD's '.tlt'
    %     file. proj_eZYZ holds the information in eulerZYZ format.
    %   SET_ANGLES(TOMO_IX,TLTFILE,XFFILE) Reads a TLTFILE and a XFFILE, 
    %     and sets up the TOMO_IX-th entry of proj_eZYZ. TLTFILE must have
    %     the IMOD's '.tlt' extension, and XFFILE, the '.xf' one. proj_eZYZ
    %     and proj_shift holds the information in eulerZYZ format and
    %     angstroms, respectively. The shifts on XF will be converted to
    %     angstroms using the current pixel size of the selected tomogram.
    %   SET_ANGLES(TOMO_IX,TLTFILE,XFFILE.XFAPIX) Reads a TLTFILE and a 
    %     XFFILE but uses XFAPIX to convert the shifts on XFFILE from
    %     pixels to angstroms.
    %
    %   See also SUSAN.Data.TomosInfo.set_stack, SUSAN.Data.TomosInfo.set_defocus.
    
        obj.proj_eZYZ(:,:,tomo_ix)   = 0;
        obj.proj_shift(:,:,tomo_ix)  = 0;
        obj.proj_weight(:,:,tomo_ix) = 0;
    
        if( nargin == 3 )
            tlt = SUSAN.IO.read_tlt(tltname);
            obj.proj_eZYZ  (1:length(tlt),2,tomo_ix) = tlt;
            obj.proj_weight(1:length(tlt),:,tomo_ix) = 1;
        elseif( nargin > 4 )
            apix_work = obj.pix_size(tomo_ix);
            if( nargin == 5 )
                apix_work = xfapix;
            end
            tlt = SUSAN.IO.read_tlt(tltname);
            xf  = SUSAN.IO.read_xf (xfname );
            for i = 1:length(tlt)
                Rtlt = eul2rotm([0 tlt(i) 0]*pi/180,'ZYZ');
                Rxf  = [xf(1,1,i) xf(1,2,i) 0; xf(2,1,i) xf(2,2,i) 0; 0 0 1]';
                Txf  = [xf(1,3,i);xf(2,3,i);0] * apix_work;
                Txf  = -Rxf*Txf;
                R    =  Rxf*Rtlt;
                obj.proj_eZYZ  (i,:,tomo_ix) = rotm2eul(R,'ZYZ')*180/pi;
                obj.proj_shift (i,:,tomo_ix) = [Txf(1) Txf(2)];
                obj.proj_weight(i,:,tomo_ix) = 1;
            end
        else
            error('Wrong number of inputs');
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_defocus(obj,tomo_ix,filename,fields)
    % SET_DEFOCUS Sets the deofcus information for a tomogram.
    %   SET_DEFOCUS(TOMO_IX,DEFFILE) Reads a DEFFILE and sets up the
    %   TOMO_IX-th entry of defocus. DEFFILE can be an IMOD's '.defocus'
    %   file, or a '.txt' file (SUSAN's CTF estimation).
    %   SET_DEFOCUS(...,FIELDS) read a few selected values from a '.txt'
    %   file (SUSAN's CTF estimation). Values:
    %    - 'Full' : Reads all the available defocus information (default).
    %    - 'Basic': Reads only U, V and the angle.
    %    - 'Env'  : Reads U, V, angle Bfactor and ExpFilter.
    %
    %   See also SUSAN.Data.TomosInfo.set_stack,
    %   SUSAN.Data.TomosInfo.set_angles, SUSAN.Modules.CtfEstimator
    
        if(nargin < 4)
            fields = 'Full';
        end
    
        if( SUSAN.Utils.is_extension(filename,'.defocus') )
            def = SUSAN.IO.read_defocus(filename);
            obj.defocus(1:size(def,1),1:3,tomo_ix) = def;
            
        elseif( SUSAN.Utils.is_extension(filename,'.txt') )
            fp = fopen(filename,'r');
            data = fscanf(fp,'%f',[8 inf]);
            fclose(fp);
            
            obj.defocus(:,:,tomo_ix) = data';
            
            if( strcmp(fields,'Basic') )
                obj.defocus(:,5:7,:) = 0;
            elseif( strcmp(fields,'Env') )
                obj.defocus(:,7,:) = 0;
            elseif( ~strcmp(fields,'Full') )
                fprintf(['Unknown field "' fields '". Valid values: Full, Basic and Env\n']);
            end
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function save(obj,filename)     
    % SAVE saves the current TomosInfo object into a file.
    %   SAVE(FILENAME) Saves the TomosInfo object in FILENAME. File
    %   extension: 'tomostxt'.
    %
    %   See also SUSAN.Data.TomosInfo, SUSAN.Data.TomosInfo.load.
        
        fp = SUSAN.Utils.TxtRW.create_file(filename,'tomostxt');

        SUSAN.Utils.TxtRW.write_pair_uint(fp,'num_tomos',size(obj.proj_eZYZ,3));
        SUSAN.Utils.TxtRW.write_pair_uint(fp,'num_projs',size(obj.proj_eZYZ,1));
        
        for i = 1:size(obj.proj_eZYZ,3)
            
            fprintf(fp,'## Tomogram/Stack %d\n',i);
            
            SUSAN.Utils.TxtRW.write_pair_uint    (fp,'tomo_id'   ,obj.tomo_id(i));
            SUSAN.Utils.TxtRW.write_pair_uint_arr(fp,'tomo_size' ,obj.tomo_size(i,:));
            SUSAN.Utils.TxtRW.write_pair_char    (fp,'stack_file',obj.stack_file{i});
            SUSAN.Utils.TxtRW.write_pair_uint_arr(fp,'stack_size',obj.stack_size(i,:));
            SUSAN.Utils.TxtRW.write_pair_single  (fp,'pix_size'  ,obj.pix_size(i));
            SUSAN.Utils.TxtRW.write_pair_single  (fp,'kv'        ,obj.voltage(i));
            SUSAN.Utils.TxtRW.write_pair_single  (fp,'cs'        ,obj.sph_aberr(i));
            SUSAN.Utils.TxtRW.write_pair_single  (fp,'ac'        ,obj.amp_contrast(i));
            SUSAN.Utils.TxtRW.write_pair_uint    (fp,'num_proj'  ,obj.num_proj(i));
            
            
            fprintf(fp,'#euler.Z  euler.Y  euler.Z  shift.X  shift.Y    weight');
            fprintf(fp,'  Defocus.U  Defocus.V  Def.ang  PhShift');
            fprintf(fp,'  BFactor  ExpFilt');
            fprintf(fp,' Res.angs FitScore\n');
            
            for j = 1:obj.num_proj(i)
                fprintf(fp,'%8.3f %8.3f %8.3f ',obj.proj_eZYZ(j,1,i),obj.proj_eZYZ(j,2,i),obj.proj_eZYZ(j,3,i));
                fprintf(fp,'%8.2f %8.2f ',obj.proj_shift(j,1,i),obj.proj_shift(j,2,i));
                fprintf(fp,'%9.4f ',obj.proj_weight(j,1,i));
                fprintf(fp,'%10.2f %10.2f ',obj.defocus(j,1,i),obj.defocus(j,2,i)); % Defocus.U    Defocus.V
                fprintf(fp,'%8.3f %8.3f '  ,obj.defocus(j,3,i),obj.defocus(j,4,i)); % Def.ang      Def.ph_shft
                fprintf(fp,'%8.2f %8.2f '  ,obj.defocus(j,5,i),obj.defocus(j,6,i)); % Def.BFactor  Def.ExpFilt
                fprintf(fp,'%8.4f %8.5f '  ,obj.defocus(j,7,i),obj.defocus(j,8,i)); % Def.max_res  Def.score
                fprintf(fp,'\n');
            end
            
        end
        
        fclose(fp);
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

methods(Access=private)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function allocate_properties(obj,num_tomos,max_proj)
        obj.tomo_id      = (1:num_tomos)';
        obj.stack_file   = cell (num_tomos,1);
        obj.stack_size   = zeros(num_tomos,3);
        obj.num_proj     = zeros(num_tomos,1);
        obj.tomo_size    = zeros(num_tomos,3);
        obj.pix_size     = ones (num_tomos,1);
        obj.proj_eZYZ    = zeros(max_proj,3,num_tomos);
        obj.proj_shift   = zeros(max_proj,2,num_tomos);
        obj.proj_weight  = zeros(max_proj,1,num_tomos);
        obj.voltage      = ones (num_tomos,1)*300;
        obj.sph_aberr    = ones (num_tomos,1)*2.7;
        obj.amp_contrast = ones (num_tomos,1)*0.07;
        obj.defocus      = zeros(max_proj,8,num_tomos);
        for i = 1:num_tomos
            obj.stack_file{i} = '';
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function internal_load(obj,filename)
        fp = SUSAN.Utils.TxtRW.open_existing_file(filename,'tomostxt');
        
        num_tomos = SUSAN.Utils.TxtRW.read_tag_int(fp,'num_tomos');
        num_projs = SUSAN.Utils.TxtRW.read_tag_int(fp,'num_projs');
        
        obj.allocate_properties(num_tomos,num_projs);
        
        for i = 1:num_tomos
            
            obj.tomo_id(i)      = SUSAN.Utils.TxtRW.read_tag_int    (fp,'tomo_id');
            obj.tomo_size(i,:)  = SUSAN.Utils.TxtRW.read_tag_int_arr(fp,'tomo_size');
            obj.stack_file{i}   = SUSAN.Utils.TxtRW.read_tag_char   (fp,'stack_file');
            obj.stack_size(i,:) = SUSAN.Utils.TxtRW.read_tag_int_arr(fp,'stack_size');
            obj.pix_size(i)     = SUSAN.Utils.TxtRW.read_tag_double (fp,'pix_size');
            obj.voltage(i)      = SUSAN.Utils.TxtRW.read_tag_double (fp,'kv');
            obj.sph_aberr(i)    = SUSAN.Utils.TxtRW.read_tag_double (fp,'cs');
            obj.amp_contrast(i) = SUSAN.Utils.TxtRW.read_tag_double (fp,'ac');
            obj.num_proj(i)     = SUSAN.Utils.TxtRW.read_tag_int    (fp,'num_proj');
            
            for j = 1:obj.num_proj(i)
                tmp = SUSAN.Utils.TxtRW.read_line(fp);
                val_list = sscanf(tmp,'%f');
                obj.proj_eZYZ(j,1:3,i)  = val_list(1:3);
                obj.proj_shift(j,1:2,i) = val_list(4:5);
                obj.proj_weight(j,1,i)  = val_list(6);
                obj.defocus(j,1:8,i)    = val_list(7:14);
            end

        end
        
        fclose(fp);
    end
    
end

end
