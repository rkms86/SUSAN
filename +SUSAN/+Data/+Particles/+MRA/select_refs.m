	
function ptcls_out = select_refs(ptcls_in,ref_idxs)     
% SELECT_REFS selects the particles that belongs to specific classes.
%   PTCLS_OUT = SELECT_REFS(PTCLS_IN,REF_IDXS) returns a subset of the
%   particles from PTCLS_IN that belongs to the classes/references
%   REF_IDXS (from 1 to NUM_REFS).

    if( ~isa(ptcls_in,'SUSAN.Data.ParticlesInfo') )
        error('First argument must be a SUSAN.Data.ParticlesInfo object.');
    end

	if( ~isvector(ref_idxs) && ~isscalar(ref_idxs) )
		error('Error in the REF_IDXS argument: it must be a scalar or a vector');
	end

	ix = zeros(size(ptcls_in.ptcl_id));
	for i = 1:length(ref_idxs)
		ix = ix | ( ptcls_in.class_cix == (ref_idxs(i)-1) );
	end

	ptcls_out = ptcls_in.select(ix);

	cix = unique(ptcls_out.class_cix);

    ptcls_out.ali_w = ptcls_out.ali_w(:,:,cix+1);
	ptcls_out.ali_t = ptcls_out.ali_t(:,:,cix+1);
	ptcls_out.ali_cc = ptcls_out.ali_cc(:,:,cix+1);
	ptcls_out.ali_eZYZ = ptcls_out.ali_eZYZ(:,:,cix+1);
    
	LUT = zeros(1,max(cix)+1);
	for i = 1:length(ref_idxs)
		LUT( ref_idxs(i) ) = i-1;
	end

	ptcls_out.class_cix(:) = LUT(ptcls_out.class_cix+1);

end




