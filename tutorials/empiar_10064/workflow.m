
rslts_time = zeros(7,1);

% 00:10:14
% 00:20:25
% 00:09:26
% 00:24:19
% 00:04:03

%%

N = 4;
K = 61;
tomos = SUSAN.Data.TomosInfo(N,K);
apix = 2.62;
tsiz = [3710 3710 880];

for i = 1:N
    tomo_base = sprintf('../tomo%02d/mixedCTEM_tomo%d',i,i);
    
    tomos.tomo_id(i) = i;
    
    tomos.set_stack (i,[tomo_base '_ali.mrc']);
    tomos.set_angles(i,[tomo_base '.tlt']);
    
    tomos.pix_size(i,:)  = apix;
    tomos.tomo_size(i,:) = tsiz;
end
tomos.save('tomos_raw_b0.tomostxt');

%%

tomos = SUSAN.read('tomos_raw_b0.tomostxt');
grid = SUSAN.Data.ParticlesInfo.grid2D(120,tomos);
grid.save('grid.ptclsraw');

%%

tic;
ctf_est = SUSAN.Modules.CtfEstimator(400);
ctf_est.gpu_list = [0 1 2 3 4 5 6];
ctf_est.resolution.min = 50;
ctf_est.resolution.max = 6;
ctf_est.defocus.min = 10000;
ctf_est.defocus.smax = 55000;
ctf_est.refine_def.range = 6000;
ctf_est.refine_def.step   = 10;
ctf_est.tlt_range = 6000;
ctf_est.verbose = 2;
tomos_ctf = ctf_est.estimate('ctf_grid','grid.ptclsraw','tomos_raw_b0.tomostxt');
tomos_ctf.defocus(:,5:7,:) = 0;
tomos_ctf.save('tomos_tlt_b0.tomostxt');
disp(toc);

%%

N = 4;
K = 61;
tomos = SUSAN.Data.TomosInfo(N,K);
apix = 2.62*2;
tsiz = [3710 3710 880]/2;

for i = 1:N
    tomo_base = sprintf('../tomo%02d/mixedCTEM_tomo%d',i,i);
    
    tomos.tomo_id(i) = i;
    
    tomos.set_stack  (i,[tomo_base '.b2.ali.mrc']);
    tomos.set_angles (i,[tomo_base '.tlt']);
    tomos.set_defocus(i,sprintf('ctf_grid/Tomo%03d/defocus.txt',i),'Basic');
    
    tomos.pix_size(i,:)  = apix;
    tomos.tomo_size(i,:) = tsiz;
end
tomos.save('tomos_tlt_b2.tomostxt');

%%

N = 4;
K = 61;
tomos = SUSAN.Data.TomosInfo(N,K);
apix = 2.62*4;
tsiz = [928 928 220];

for i = 1:N
    tomo_base = sprintf('../tomo%02d/mixedCTEM_tomo%d',i,i);
    
    tomos.tomo_id(i) = i;
    
    tomos.set_stack  (i,[tomo_base '.b4.ali.mrc']);
    tomos.set_angles (i,[tomo_base '.tlt']);
    tomos.set_defocus(i,sprintf('ctf_grid/Tomo%03d/defocus.txt',i),'Basic');
    
    tomos.pix_size(i,:)  = apix;
    tomos.tomo_size(i,:) = tsiz;
end
tomos.save('tomos_tlt_b4.tomostxt');

%%

tomos = SUSAN.read('tomos_tlt_b4.tomostxt');
tbl = SUSAN.read('picked_b8.tbl');
tbl(:,[24 25 26]) = 2*tbl(:,[24 25 26]);
tbl(:,[7 8 9]) = 360*rand(size(tbl(:,[7 8 9])));
ptcls = SUSAN.Data.ParticlesInfo(tbl,tomos);
ptcls.save('prj_001.ptclsraw');

%%

avgr = SUSAN.Modules.Averager;
avgr.gpu_list = [0 1 2 3 4 5 6];
avgr.set_ctf_correction('phase_flip');
avgr.rec_halves = false;
avgr.reconstruct('prj_001','tomos_tlt_b4.tomostxt','prj_001.ptclsraw',64);

%%

ptcls = SUSAN.read('prj_001.ptclsraw');
SUSAN.Data.Particles.MRA.duplicate(ptcls,1);
ptcls.save('prj_001.ptclsraw');

%%

SUSAN.IO.write_mrc( -dynamo_sphere(4,64,[33 33 33],2), 'gb_b4.mrc', 2.62*4 );
SUSAN.IO.write_mrc( dynamo_sphere(21,64,[33 33 33],3), 'mask_sph_b4.mrc', 2.62*4 );

%%

refs = SUSAN.Data.ReferenceInfo.create(2);
refs(1).map  = 'emd_3420_b4.mrc';
refs(1).mask = 'mask_sph_b4.mrc';
refs(2).map  = 'gb_b4.mrc';
refs(2).mask = 'mask_sph_b4.mrc';
refs.save(refs,'prj_001_b4.refstxt');

%%

mngr = SUSAN.Project.Manager('prj_001_b4',64);

mngr.initial_reference = 'prj_001_b4.refstxt';
mngr.initial_particles = 'prj_001.ptclsraw';
mngr.tomogram_file     = 'tomos_tlt_b4.tomostxt';

mngr.gpu_list = [0 1 2 3 4 5 6];

mngr.aligner.set_ctf_correction('on_reference');
mngr.aligner.set_normalization('zm1s');
mngr.aligner.padding = 0;
mngr.aligner.drift = true;
mngr.averager.set_ctf_correction('phase_flip');

%

start_time = tic;
%% Iteration 1   17:59:20

for i = 1
    mngr.cc_threshold = 0.9;
    mngr.alignment_type = 3;
    mngr.aligner.set_angular_search(360,30,360,30);
    mngr.aligner.set_angular_refinement(2,1);
    mngr.aligner.set_offset_ellipsoid(24,1);
    mngr.aligner.bandpass.highpass = 0;
    mngr.aligner.bandpass.lowpass = 21;
    mngr.execute_iteration(i);
end

%

for i = 2
    mngr.cc_threshold = 0.9;
    mngr.alignment_type = 3;
    mngr.aligner.set_angular_search(0,1,0,1);
    mngr.aligner.set_angular_refinement(0,1);
    mngr.aligner.set_offset_ellipsoid(12,1);
    mngr.aligner.bandpass.highpass = 0;
    mngr.aligner.bandpass.lowpass = 21;
    mngr.execute_iteration(i);
end

%%

for i = 3:7
    mngr.cc_threshold = 0.9;
    mngr.alignment_type = 3;
    mngr.aligner.set_angular_search(32,6,32,6);
    mngr.aligner.set_angular_refinement(2,1);
    mngr.aligner.set_offset_ellipsoid(20,1);
    mngr.aligner.bandpass.highpass = 0;
    mngr.aligner.bandpass.lowpass = 21;
    mngr.execute_iteration(i);
end

%%

end_time = toc(start_time);
disp('Total time:');
disp(end_time);
disp(datestr(seconds(end_time),'HH:MM:SS'));
rslts_time(1) = (end_time);

%%

p_rslt = mngr.get_ptcls(7);
p_class1 = SUSAN.Data.Particles.MRA.select_refs(p_rslt,1);
p_class1.position = p_class1.position + p_class1.ali_t;
p_class1.ali_t(:) = 0;
p_class1_exc = SUSAN.Data.Particles.Geom.discard_closer(p_class1,20);
pw = p_class1_exc.select(p_class1_exc.ali_cc<0.9&p_class1_exc.ali_cc>0.01);
pw.halfsets_by_Y();
pw.save('prj_002.ptclsraw');
%%

avgr = SUSAN.Modules.Averager;
avgr.gpu_list = [0 1 2 3 4 5 6];
avgr.set_ctf_correction('phase_flip');
avgr.rec_halves = false;
avgr.reconstruct('prj_002','tomos_tlt_b2.tomostxt','prj_002.ptclsraw',128);

%%

h = dynamo_sphere(5,12); h = h/sum(h(:));
a = convn( single(dbandpass('prj_002_class001.mrc',[0 20 2])<-1), h, 'same' );
h = dynamo_sphere(4,10); h = h/sum(h(:));
b = convn( single(a>0.05), h, 'same' );
dwrite(b,'mask_loose_b2.mrc');

%%

refs = SUSAN.Data.ReferenceInfo.create(1);
refs(1).map  = 'prj_002_class001.mrc';
refs(1).mask = 'mask_loose_b2.mrc';
refs.save(refs,'prj_002_b2.refstxt');

%%

mngr = SUSAN.Project.Manager('prj_002_b2',128);

mngr.initial_reference = 'prj_002_b2.refstxt';
mngr.initial_particles = 'prj_002.ptclsraw';
mngr.tomogram_file     = 'tomos_tlt_b2.tomostxt';

mngr.gpu_list = [0 1 2 3 4 5 6];

mngr.aligner.set_ctf_correction('on_reference');
mngr.aligner.set_normalization('zm1s');
mngr.aligner.padding = 0;
mngr.aligner.drift = true;
mngr.averager.set_ctf_correction('phase_flip');

%%

start_time = tic;

% Iteration 1

for i = 1
    mngr.cc_threshold = 0.95;
    mngr.alignment_type = 3;
    mngr.aligner.set_angular_search(0,1,0,1);
    mngr.aligner.set_angular_refinement(0,1);
    mngr.aligner.set_offset_ellipsoid(10,1);
    mngr.aligner.bandpass.highpass = 0;
    mngr.aligner.bandpass.lowpass = 30;
    lp = mngr.execute_iteration(i);
end

%

for i = 2:7
    as = atan2d(1,lp);
    mngr.cc_threshold = 0.6;
    mngr.alignment_type = 3;
    mngr.aligner.set_angular_search(4*as,as,4*as,as);
    mngr.aligner.set_angular_refinement(2,2);
    mngr.aligner.set_offset_ellipsoid(6,0.5);
    mngr.aligner.bandpass.highpass = 0;
    mngr.aligner.bandpass.lowpass = lp;
    bp = mngr.execute_iteration(i);
    lp = min(bp,lp+1);
end

%%

for i = 8:10
    as = atan2d(1,lp);
    mngr.cc_threshold = 0.95;
    mngr.alignment_type = 2;
    mngr.aligner.set_angular_search(4*as,as,4*as,as);
    mngr.aligner.set_angular_refinement(1,1);
    mngr.aligner.set_offset_ellipsoid(6,0.5);
    mngr.aligner.bandpass.highpass = 0;
    mngr.aligner.bandpass.lowpass = lp;
    bp = mngr.execute_iteration(i);
    lp = min(bp,lp+2);
end

%%

end_time = toc(start_time);
disp('Total time:');
disp(end_time);
disp(datestr(seconds(end_time),'HH:MM:SS'));
rslts_time(2) = (end_time);

%%

h = dynamo_sphere(9,20); h = h/sum(h(:));
a = convn( single(dbandpass('prj_002_b2/ite_0010/map_class001.mrc',[0 25 2])<-1), h, 'same' );
h = dynamo_sphere(6,14); h = h/sum(h(:));
b = convn( single(a>0.1), h, 'same' );
dwrite(b,'mask_loose_b2_2.mrc');

%%

ptcls = mngr.get_ptcls(10);
ptcls.position = ptcls.position + ptcls.ali_t;
ptcls.ali_t(:) = 0;
pw = ptcls.select( ptcls.ali_cc > 0.003 );
pw.class_cix(:) = floor( 2*rand(size(pw.class_cix)) );
SUSAN.Data.Particles.MRA.duplicate(pw,1);
pw.save('prj_003.ptclsraw');

%%

avgr = SUSAN.Modules.Averager;
avgr.gpu_list = [0 1 2 3 4 5 6];
avgr.set_ctf_correction('wiener_ssnr',0.75,0.2);
avgr.rec_halves = false;
avgr.reconstruct('prj_003','tomos_tlt_b2.tomostxt','prj_003.ptclsraw',128);

%%

refs = SUSAN.Data.ReferenceInfo.create(2);
refs(1).map  = 'prj_003_class001.mrc';
refs(1).mask = 'mask_loose_b2_2.mrc';
refs(2).map  = 'prj_003_class002.mrc';
refs(2).mask = 'mask_loose_b2_2.mrc';
refs.save(refs,'prj_003_b2.refstxt');

%%

mngr = SUSAN.Project.Manager('prj_003_b2',128);

mngr.initial_reference = 'prj_003_b2.refstxt';
mngr.initial_particles = 'prj_003.ptclsraw';
mngr.tomogram_file     = 'tomos_tlt_b2.tomostxt';

mngr.gpu_list = [0 1 2 3 4 5 6];

mngr.aligner.set_ctf_correction('on_reference');
mngr.aligner.set_normalization('zm1s');
mngr.aligner.padding = 0;
mngr.aligner.drift = true;
mngr.averager.set_ctf_correction('wiener_ssnr',0.75,0.2);

%%

start_time = tic;

% Iteration 1

for i = 1:5
    mngr.cc_threshold = 0.4;
    mngr.alignment_type = 3;
    mngr.aligner.set_angular_search(4,1,4,1);
    mngr.aligner.set_angular_refinement(0,1);
    mngr.aligner.set_offset_ellipsoid(10,1);
    mngr.aligner.bandpass.highpass = 0;
    mngr.aligner.bandpass.lowpass = 48;
    mngr.execute_iteration(i);
end

%

end_time = toc(start_time);
disp('Total time:');
disp(end_time);
disp(datestr(seconds(end_time),'HH:MM:SS'));
rslts_time(3) = (end_time);

%%

p_rslt = mngr.get_ptcls(5);
p_class1 = SUSAN.Data.Particles.MRA.select_refs(p_rslt,2);
p_class1.position = p_class1.position + p_class1.ali_t;
p_class1.ali_t(:) = 0;
p_class1_exc = SUSAN.Data.Particles.Geom.discard_closer(p_class1,20);
pw = p_class1_exc.select(p_class1_exc.ali_cc<0.03);
pw.halfsets_by_Y();
pw.save('prj_004.ptclsraw');

%%  

ptmp = SUSAN.read('prj_004.ptclsraw');
pw = SUSAN.Data.Particles.Geom.discard_closer(ptmp,50);
pw.save('grid_z.ptclsraw');

%%

%tic;
ctf_est = SUSAN.Modules.CtfEstimator(400);
ctf_est.gpu_list = [0 1 2 3 4 5 6];
ctf_est.resolution.min = 40;
ctf_est.resolution.max = 7.5;
ctf_est.defocus.min = 10000;
ctf_est.defocus.max = 55000;
ctf_est.refine_def.range = 6000;
ctf_est.refine_def.step   = 10;
ctf_est.tlt_range = 6000;
ctf_est.verbose = 2;
tomos_ctf = ctf_est.estimate('ctf_rslt','grid_z.ptclsraw','tomos_tlt_b0.tomostxt');
tomos_ctf.defocus(:,5:7,:) = 0;
tomos_ctf.save('tomos_ctf_b0.tomostxt');
%disp(toc);

%%

tomos_ctf = SUSAN.read('tomos_ctf_b0.tomostxt');
p_class1_exc.update_defocus(tomos_ctf);
p_class1_exc.defocus(:,5:7,:) = 0;
p_class1_exc.save('prj_004.ptclsraw');

%%

avgr = SUSAN.Modules.Averager;
avgr.gpu_list = [0 1 2 3 4 5 6];
avgr.set_ctf_correction('wiener_ssnr',0.75,0.2);
avgr.rec_halves = true;
avgr.reconstruct('prj_004','tomos_ctf_b0.tomostxt','prj_004.ptclsraw',256);

%%

h = dynamo_sphere(9,20); h = h/sum(h(:));
a = convn( single(dbandpass('prj_004_class001.mrc',[0 25 2])<-1), h, 'same' );
h = dynamo_sphere(6,14); h = h/sum(h(:));
b = convn( single(a>0.05), h, 'same' );
dwrite(b,'mask_loose_b1.mrc');

%%

refs = SUSAN.Data.ReferenceInfo.create(1);
refs(1).map  = 'prj_004_class001.mrc';
refs(1).mask = 'mask_loose_b1.mrc';
refs.save(refs,'prj_004_b1.refstxt');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mngr = SUSAN.Project.Manager('prj_004_b1',256);

mngr.initial_reference = 'prj_004_b1.refstxt';
mngr.initial_particles = 'prj_004.ptclsraw';
mngr.tomogram_file     = 'tomos_ctf_b0.tomostxt';

mngr.gpu_list = [0 1 2 3 4 5 6];

mngr.align_references = true;

mngr.aligner.set_ctf_correction('on_reference');
mngr.aligner.set_normalization('zm1s');
mngr.aligner.padding = 0;
mngr.aligner.drift = true;
mngr.averager.set_ctf_correction('phase_flip',0.75,0.2);
mngr.averager.set_normalization('zm1s');

%%

start_time = tic;

%% Iteration 1

lp = 45;
for i = 1
    mngr.cc_threshold = 0.95;
    mngr.alignment_type = 3;
    mngr.aligner.set_angular_search(0,1,0,1);
    mngr.aligner.set_angular_refinement(0,1);
    mngr.aligner.set_offset_ellipsoid(8,0.5);
    mngr.aligner.bandpass.highpass = 0;
    mngr.aligner.bandpass.lowpass = lp;
    mngr.execute_iteration(i);
end

%% Iteration 2

for i = 2:5
    mngr.cc_threshold = 0.95;
    mngr.alignment_type = 3;
    mngr.aligner.set_angular_search(6.8,0.85,6.8,0.85);
    mngr.aligner.set_angular_refinement(2,1);
    mngr.aligner.set_offset_ellipsoid(6,0.5);
    mngr.aligner.bandpass.highpass = 0;
    mngr.aligner.bandpass.lowpass = 50;
    mngr.execute_iteration(i);
end

%%

end_time = toc(start_time);
disp('Total time:');
disp(end_time);
disp(datestr(seconds(end_time),'HH:MM:SS'));
rslts_time(4) = (end_time);

%%

tomos_ctf = SUSAN.read('tomos_ctf_b0.tomostxt');
tomos_ctf.defocus(:,6,:) = 8.*abs(tomos_ctf.proj_eZYZ(:,2,:)/2);
ptcls = mngr.get_ptcls(5);
ptcls.update_defocus(tomos_ctf);
p1 = ptcls.select(ptcls.ali_cc<11e-3&ptcls.ali_cc>5.5e-3);
p1.save('prj_005_b1.ptclsraw');

%%

avgr = SUSAN.Modules.Averager;
avgr.gpu_list = [0 1 2 3 4 5 6];
avgr.set_normalization('zm1s')
avgr.set_ctf_correction('wiener');
avgr.rec_halves = true;
avgr.bandpass.rolloff = 6;
avgr.reconstruct('prj_005_b1','tomos_ctf_b0.tomostxt','prj_005_b1.ptclsraw',256);


