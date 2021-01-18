%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function enable_by_tilt(ptcls_in,tomos_list,ang)
% ENABLE_BY_TILT enables/disables projections by their tilt angle.
%   ENABLE_BY_TILT(PTCLS_IN,TOMOS_LIST,ANG) enables/disables the
%   projections on PTCLS_IN according the tilt angles from TOMO_LIST.
%   Enables the projections with tilts smaller than ANG, disables the rest.
%
%   See also SUSAN.Data.ParticlesInfo.

    if( ~isa(ptcls_in,'SUSAN.Data.ParticlesInfo') )
        error('First argument must be a SUSAN.Data.ParticlesInfo object.');
    end

    if( ~isa(tomos_list,'SUSAN.Data.TomosInfo') )
        error('Second argument must be a SUSAN.Data.TomosInfo object.');
    end

    if( ~(ang>0) )
        error('Third argument must be > 0.');
    end

    ptcls_in.prj_w = ParticlesGeom_enable_by_tilt_angle(ptcls_in.tomo_cix,tomos_list.proj_eZYZ,single(ang));
    ptcls_in.prj_w = ptcls_in.prj_w.*tomos_list.proj_weight(:,:,ptcls_in.tomo_cix+1);
	
end


