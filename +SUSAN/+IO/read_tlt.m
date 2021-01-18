function tlt = read_tlt(tltname)
% READ_TLT Reads a IMOD's tlt file.
%   TLT = READ_TLT(TLTNAME) Reads TLTNAME as an array of angles.

fp = fopen(tltname,'r');
tlt = fscanf(fp,'%f');
fclose(fp);

end
