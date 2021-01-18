function write_mrc(map,filename,apix)
% Writes a MRC file.
%   WRITE_MRC(MAP,FILENAME) Saves MAP in FILENAME using the MRC format.
%   WRITE_MRC(...,APIX) Additionally, it writes the pixel size in the file.


if( nargin < 3 )
    apix = 1;
end

fp = fopen(filename,'w+');

% Empty headers
fwrite(fp,zeros(1,1024,'uint8'));

% Save content
fwrite(fp,single(map),'float32');

% Save size and Mode:
fseek(fp, 0, 'bof');
buffer = zeros(1,4,'uint32');
buffer(:) = [uint32(size(map)), 2];
fwrite(fp,buffer,'uint32');

% Save Pixel Size:
xyz_len = single(size(map));
xyz_len = apix*xyz_len;
fseek(fp, 28, 'bof');
buffer = zeros(1,3,'uint32');
buffer(:) = uint32(size(map));
fwrite(fp,buffer,'uint32');
fseek(fp, 40, 'bof');
fwrite(fp,xyz_len,'float32');

% Save Mapping
fseek(fp, 64, 'bof');
buffer = zeros(1,3,'uint32');
buffer(:) = [1 2 3];
fwrite(fp,buffer,'uint32');

% finished
fclose(fp);


end


