function [mrc_size,mrc_mode,mrc_apix] = get_mrc_info(filename)
% GET_MRC_INFO reads the info from header of an MRC file.
%   [MRC_SIZE, MRC_MODE, MRC_APIX] = GET_MRC_INFO(FILENAME) reads
%   the header of FILENAME, assuming is a MRC file, and returns
%   the size, mode and pixel size in MRC_SIZE, MRC_MODE and
%   MRC_APIX, respectively.


fp = fopen(filename,'rb');
mrc_size = fread(fp,[1,3],'uint32');

fseek(fp,12,'bof');
mrc_mode = fread(fp,[1,1],'uint32');

fseek(fp,28,'bof');
mx = fread(fp,[1,1],'uint32');
fseek(fp,40,'bof');
xlen = fread(fp,[1,1],'single');

if( xlen == 0 )
    mrc_apix = 1;
else
    mrc_apix = xlen/mx;
end

fclose(fp);

end
