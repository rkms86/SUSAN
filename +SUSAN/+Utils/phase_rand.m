function v_rand = phase_rand(arg1, fpix)
% PHASE_RAND randomize the phase of a volume.
%   VOUT = PHASE_RAND(V,FPIX)

v = conditional_load_mrc(arg1);
m = ifftshift(1-SUSAN.Utils.mask_sphere(fpix,size(v,1)));

V = fftn(v);
A = abs(V);
phi = angle(V) + 2*pi*m.*rand(size(V));

v_rand = ifftn( A.*exp(1i*phi) );

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function v = conditional_load_mrc(arg)

if( ischar(arg) )
    v = SUSAN.IO.read_mrc(arg);
else
    v = arg;
end

end



