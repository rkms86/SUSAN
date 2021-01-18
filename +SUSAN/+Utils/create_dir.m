function create_dir(dirname)
    if( ~exist(dirname,'dir') )
        mkdir(dirname);
    end
end