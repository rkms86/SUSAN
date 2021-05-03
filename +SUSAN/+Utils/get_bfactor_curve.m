function rslt = get_bfactor_curve(bfactor,apix,box_size)

R = box_size/2 + 1;
r = 1:R;
s = r/(R*apix);

rslt = exp( -s.*s*bfactor/4);

end