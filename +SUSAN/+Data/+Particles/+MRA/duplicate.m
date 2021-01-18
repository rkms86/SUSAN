
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function duplicate(ptcls_in,ref_ix)
% DUPLICATE Copies a reference alignment info and place it at the end.
%   DUPLICATE(PTCLS,REF_IX) Copies the alignment information for the class/
%   reference REF_IX and appends it to the alignment matrices.

    if( ~isa(ptcls_in,'SUSAN.Data.ParticlesInfo') )
        error('First argument must be a SUSAN.Data.ParticlesInfo object.');
    end
    
    if( nargin < 2 )
        ref_ix = 1;
    end

    ptcls_in.ali_eZYZ = cat(3, ptcls_in.ali_eZYZ, ptcls_in.ali_eZYZ(:,:,ref_ix));
    ptcls_in.ali_t    = cat(3, ptcls_in.ali_t   , ptcls_in.ali_t   (:,:,ref_ix));
    ptcls_in.ali_cc   = cat(3, ptcls_in.ali_cc  , ptcls_in.ali_cc  (:,:,ref_ix));

end
