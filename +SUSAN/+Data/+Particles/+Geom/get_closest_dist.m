%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function min_dist = get_closest_dist(obj)     
% GET_CLOSEST_DIST returns a list with the minimum distance between
% particles.
%   MIN_DIST = GET_CLOSEST_DIST() returns an array with the minimum
%   distance of each particle to the other ones.
%
%   See also SUSAN.Data.ParticlesInfo.

    if( ~isa(ptcls_in,'SUSAN.Data.ParticlesInfo') )
        error('First argument must be a SUSAN.Data.ParticlesInfo object.');
    end
    
    min_dist = [];

    tomos_id = unique( obj.tomo_id );

    for i = 1:length(tomos_id)
        t_idx = (obj.tomo_id == tomos_id(i));
        data = [obj.position(t_idx,:)+obj.ali_t(t_idx,:)];
        cur_dist = ParticlesGeom_calc_min_dist(data');
        min_dist = [min_dist; cur_dist];
    end
end
