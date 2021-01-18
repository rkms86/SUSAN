function ptcls_out = discard_closer(ptcls_in,min_dist_angs)     
% DISCARD_CLOSER discards particles too close to another ones.
%   PTCLS_OUT = DISCARD_CLOSER(PTCLS_IN,MIN_DIST) returns a copy of
%   PTCLS_IN discarding the particles that are closer than MIN_DIST (in
%   angstroms), keeping the ones with higher CC.
%
%   See also SUSAN.Data.ParticlesInfo.

    if( ~isa(ptcls_in,'SUSAN.Data.ParticlesInfo') )
        error('First argument must be a SUSAN.Data.ParticlesInfo object.');
    end
    
    if( ~(min_dist_angs>0) )
        error('Second argument must be > 0.');
    end
    
    idx = [];

    tomos_id = unique( ptcls_in.tomo_id );
    fprintf('    %d tomograms detected. Processing:\n',length(tomos_id));

    for i = 1:length(tomos_id)
        t_idx = (ptcls_in.tomo_id == tomos_id(i));
        n = sum(t_idx);
        data = [ptcls_in.position(t_idx,:)+ptcls_in.ali_t(t_idx,:,1) ptcls_in.ali_cc(t_idx,:) (0:(n-1))'];
        data = sortrows(data,4,'descend');
        cur_idx = ParticlesGeom_select_min_dist(data',single(min_dist));
        idx = [idx; cur_idx];
        fprintf('        Tomogram %d: from %7d to %7d particles\n',tomos_id(i),n,sum(cur_idx));
    end

    ptcls_out = ptcls_in.select(idx>0);
end
