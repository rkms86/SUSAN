function [map, apix] = read_mrc(filename)
% READ_MRC Reads a MRC file.
%   [MAP, APIX] = READ_MRC(FILENAME) Reads FILENAME as a MRC map and
%   returns it in MAP. Additionally, it can return the pixel size stored in
%   the file in APIX.

if( ~exist(filename,'file') )
    error('File %s does not exist.',filename);
end

fp = fopen(filename,'rb');

% Read Size, Mode and Mapping of the MRC. [XYZ] is the supported mapping.
fseek(fp, 0, 'bof');
stack_size = fread(fp,3,'uint32')';
stack_mode = fread(fp,1,'uint32');
fseek(fp, 64, 'bof');
stack_map = fread(fp,3,'uint32');
if( sum( abs( stack_map-[1;2;3] ) ) ~= 0 )
    error('Unsupported mapping in the MRC file %s',filename);
end

% Read pixel size, if required.
if( nargout > 1 )
    fseek(fp, 28, 'bof');
    mx   = fread(fp,1,'uint32');
    fseek(fp, 40, 'bof');
    xlen = fread(fp,1,'float32');

    if( xlen == 0 ) 
        apix = 1;
    else
        apix = xlen/mx;
    end
end

% Additional offset
fseek(fp, 92, 'bof');
offset = fread(fp,1,'uint32');

% Read data according to the MODE
fseek(fp, 1024+offset, 'bof');
if    ( stack_mode == 0 ) % 8-bit signed integer (range -128 to 127)
    map = single(fread(fp,inf,'int8'));
elseif( stack_mode == 1 ) % 16-bit signed integer
    map = single(fread(fp,inf,'int16'));
elseif( stack_mode == 2 ) % 32-bit real
    map = single(fread(fp,inf,'float32'));
elseif( stack_mode == 6 ) % 16-bit unsigned integer
    map = single(fread(fp,inf,'uint16'));
else
    error('s: MRC file with unsupported mode %d',filename,stack_mode);
end
fclose(fp);

% Validate size:
if( numel( map ) == (stack_size(1)*stack_size(2)*stack_size(3)) )
    map = reshape(map,stack_size );
else
    error('%s: Corrupted file',filename);

end
