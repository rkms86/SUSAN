function th = unimodal_th(pt_cloud,v1,v2)

v1 = repmat(v1,size(pt_cloud,1),1);
v2 = repmat(v2,size(pt_cloud,1),1);

a = v1 - v2;
b = pt_cloud - v2;

a(end,3) = 0;
b(end,3) = 0;

d = sqrt(sum(cross(a,b,2).^2,2)) ./ sqrt(sum(a.^2,2));

[~,ix_max] = max(d);

th = pt_cloud(ix_max,1);

end
