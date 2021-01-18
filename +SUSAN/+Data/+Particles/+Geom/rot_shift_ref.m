function rot_shift_ref(ptcls_in,euZYZ,t,ref)
% ROT_SHIFT_REF Applies a rotation and a shift to a reference.
%   ROT_SHIFT_REF(PTCLS,R,T,REF) Aplies the rotation R and then the shift T
%   to the reference/class REF of the particles PTCLS_IN. R can be a 3D
%   rotation matrix or euler angles in the ZYZ format. T is a 3 elements
%   vector.

    if( ~isa(ptcls_in,'SUSAN.Data.ParticlesInfo') )
        error('First argument must be a SUSAN.Data.ParticlesInfo object.');
    end

    if( nargin < 4 )
        ref = 1;
    end

    if( numel(t) ~= 3 || ~isvector(t) )
        error('Wrong format for the shift/translation argument.');
    end

    if( numel(euZYZ) == 3 )
        R = eul2rotm(euZYZ*pi/180, 'ZYZ');
    elseif( numel(euZYZ) == 9 )
        R = euZYZ;
    else
        error('Wrong format for the rotation argument.');
    end


    if( ref < 1 || ref > ptcls_in.n_refs )
        error('Invalid requested reference/class.');
    end

    [tmp_e,tmp_t] = ParticlesGeom_rot_shift(single(R),single(t),ptcls_in.ali_eZYZ(:,:,ref),ptcls_in.ali_t(:,:,ref));

    ptcls_in.ali_eZYZ(:,:,ref_ix) = tmp_e;
    ptcls_in.ali_t   (:,:,ref_ix) = tmp_t;

end
