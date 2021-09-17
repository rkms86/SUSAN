classdef ReferenceInfo < handle
% SUSAN.Data.ReferenceInfo Holds the information of the references.
%    Holds the information of the references used for the alignement
%    procedures.
%
% SUSAN.Data.ReferenceInfo Properties:
%    map  - filename of the reference map.
%    mask - filename of the mask associated to the map.
%    h1   - [Optional] filename of the first half map.
%    h2   - [Optional] filename of the second half map.
%
% SUSAN.Data.ReferenceInfo Static Methods:
%    load   - loads a ReferenceInfo array from a file.
%    save   - saves a ReferenceInfo array into a file.
%    create - creates an empty ReferenceInfo array.
%    show   - shows a ReferenceInfo array.
%    Note: The file extension for a ReferenceInfo file is 'refstxt'. The
%    file is a normal text file.

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

properties
    map  = '';
    mask = '';
    h1   = '';
    h2   = '';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

methods(Static)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function ref_list = load(ref_list_file)
    % LOAD [Static] reads a ReferenceInfo array from a file
    %   REF_LIST = SUSAN.Data.ReferenceInfo.LOAD(REF_LIST_FILE) reads file 
    %   REF_LIST_FILE and returns REF_LIST. File extension: 'refstxt'.
    %
    %   See also SUSAN.Data.ReferenceInfo.save.
        
        fp = SUSAN.Utils.TxtRW.open_existing_file(ref_list_file,'refstxt');
    
        num_refs = SUSAN.Utils.TxtRW.read_tag_int(fp,'num_ref');
        
        ref_list = SUSAN.Data.ReferenceInfo.create( num_refs );
        
        for i = 1:num_refs
            
            ref_list(i).map  = SUSAN.Utils.TxtRW.read_tag_char(fp,'map');
            ref_list(i).mask = SUSAN.Utils.TxtRW.read_tag_char(fp,'mask');
            ref_list(i).h1   = SUSAN.Utils.TxtRW.read_tag_char(fp,'h1');
            ref_list(i).h2   = SUSAN.Utils.TxtRW.read_tag_char(fp,'h2');

        end
        
        fclose(fp);
        
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function save(ref_list,ref_list_file)
    % SAVE [Static] writes a ReferenceInfo array to a file
    %   SUSAN.Data.ReferenceInfo.SAVE(REF_LIST,REF_LIST_FILE) writes array
    %   REF_LIST in the file REF_LIST_FILE. Adds extension 'refstxt' if not
    %   provided.
    %
    %   See also SUSAN.Data.ReferenceInfo.load.
    
        fp = SUSAN.Utils.TxtRW.create_file(ref_list_file,'refstxt');

        SUSAN.Utils.TxtRW.write_pair_uint(fp,'num_ref',length(ref_list));
        
        for i = 1:length(ref_list)
            
            fprintf(fp,'## Reference %d\n',i);
            
            if( ~SUSAN.Utils.TxtRW.write_pair_char(fp,'map',ref_list(i).map) )
                error('Map filename on entry %d is empty.',i);
            end
            
            if( ~SUSAN.Utils.TxtRW.write_pair_char(fp,'mask',ref_list(i).mask) )
                error('Mask filename on entry %d is empty.',i);
            end
            
            SUSAN.Utils.TxtRW.write_optional_pair_char(fp,'h1',ref_list(i).h1);
            SUSAN.Utils.TxtRW.write_optional_pair_char(fp,'h2',ref_list(i).h2);
            
        end
        
        fclose(fp);
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function ref_list = create(num_ref)
    % CREATE [Static] Creates an empty ReferenceInfo array
    %   SUSAN.Data.ReferenceInfo.CREATE(NUM_REF) Creates an empty array of 
    %   NUM_REF elements of class SUSAN.Data.ReferenceInfo.
    %
    %   See also SUSAN.Data.ReferenceInfo.load,
    %   SUSAN.Data.ReferenceInfo.save.
        
        ref_list(1:num_ref) = SUSAN.Data.ReferenceInfo;
        for i = 1:num_ref
            ref_list(i) = SUSAN.Data.ReferenceInfo;
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function show(ref_list)
    % SHOW [Static] shows the ReferenceInfo array
    %   SUSAN.Data.ReferenceInfo.SHOW(REF_LIST) Shows all the information
    %   of the SUSAN.Data.ReferenceInfo array.
        
        fprintf('Number of references: %d\n',length(ref_list));
        for i = 1:length(ref_list)
            fprintf('  Reference %d:\n',i);
            fprintf('    Map:   %s\n',ref_list(i).map );
            fprintf('    Mask:  %s\n',ref_list(i).mask);
            if( ~isempty(ref_list(i).h1) )
                fprintf('    Half1: %s\n',ref_list(i).h1  );
            end
            if( ~isempty(ref_list(i).h2) )
                fprintf('    Half2: %s\n',ref_list(i).h2  );
            end
        end
        
    end
    
end

end
