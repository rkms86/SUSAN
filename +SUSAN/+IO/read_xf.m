function xf = read_xf(xfname)
% READ_XF Reads a IMOD's xf file.
%   xf = READ_XF(XFNAME) Reads XFNAME as a collection 2D affine matrices.

fp = fopen(xfname,'r');
data = fscanf(fp,'%f',[6 inf]);
fclose(fp);

xf = zeros(2,3,size(data,2));

for i = 1:size(data,2)
    xf(:,:,i) = [data(1,i) data(2,i) data(5,i); data(3,i) data(4,i) data(6,i);];
end

end
