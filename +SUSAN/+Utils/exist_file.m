function rslt = exist_file(filename)
    if( exist(filename,'file') )
        rslt = true;
    else
        rslt = false;
    end
end