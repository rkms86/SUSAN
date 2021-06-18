function ptcls_out = expand_ptcls(ptcls_in,arr_euZYZ,arr_shifts)     
% EXPAND_PTCLS expand the particles according to a set of transformations.
%   PTCLS_OUT = EXPAND_PTCLS(PTCLS_IN,EUZYZ,SHIFTS) expand the number of 
%   particles by creating new entries using the angles in ARR_EUZYZ and the
%   shifts on ARR_SHIFTS.
	
	if( size( arr_shifts, 2 ) ~= 3 || size( arr_euZYZ, 2 ) ~= 3 || size( arr_shifts, 1 ) ~= size( arr_euZYZ, 1 ) )
		error('Wrong input dimensions');
	end

	if( ptcls_in.n_refs > 1 )
		error('Operation valid only when 1 class is available');
	end
	
	exp_n = size(arr_shifts,1);
	
	full_ix = repmat( (1:ptcls_in.n_ptcls), [exp_n 1] );
	ptcls_out = ptcls_in.select( full_ix(:) );
	ptcls_out.ptcl_id(:) = 1:ptcls_out.n_ptcls;
	
	for j = 1:exp_n
		
		R = eul2rotm(arr_euZYZ(j,:)*pi/180, 'ZYZ');
		
		[tmp_e,tmp_t] = ParticlesGeom_rot_shift(single(R),single(arr_shifts(j,:)),ptcls_in.ali_eZYZ,ptcls_in.ali_t);
        
		ptcls_out.ali_eZYZ(j:exp_n:end,:) = tmp_e;
		ptcls_out.ali_t   (j:exp_n:end,:) = tmp_t;
		ptcls_out.ali_cc  (j:exp_n:end,:) = ptcls_in.ali_cc;
	end
end
