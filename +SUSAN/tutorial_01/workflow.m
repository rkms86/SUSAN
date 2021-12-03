%% Create tomostxt BIN1 (unbin):

N = 4;
K = 61;
tomos = SUSAN.Data.TomosInfo(N,K);
apix = 2.62;
tsiz = [3710 3710 880];

for i = 1:N
    tomo_base = sprintf('data/mixedCTEM_tomo%d',i);
    
    tomos.tomo_id(i) = i;
    
    tomos.set_stack (i,[tomo_base '.b1.ali.mrc']);
    tomos.set_angles(i,[tomo_base '.tlt']);
    
    tomos.set_defocus(i,[tomo_base '.defocus']);
    
    tomos.pix_size(i,:)  = apix;
    tomos.tomo_size(i,:) = tsiz;
end
tomos.save('tomos_b1.tomostxt');

%% Create tomostxt BIN2:

N = 4;
K = 61;
tomos = SUSAN.Data.TomosInfo(N,K);
apix = 2.62*2;
tsiz = [3710 3710 880]/2;

for i = 1:N
    tomo_base = sprintf('data/mixedCTEM_tomo%d',i);
    
    tomos.tomo_id(i) = i;
    
    tomos.set_stack (i,[tomo_base '.b2.ali.mrc']);
    tomos.set_angles(i,[tomo_base '.tlt']);
    
    tomos.set_defocus(i,[tomo_base '.defocus']);
    
    tomos.pix_size(i,:)  = apix;
    tomos.tomo_size(i,:) = tsiz;
end
tomos.save('tomos_b2.tomostxt');

%% Create tomostxt BIN4:

N = 4;
K = 61;
tomos = SUSAN.Data.TomosInfo(N,K);
apix = 2.62*4;
tsiz = [3710 3710 880]/4;

for i = 1:N
    tomo_base = sprintf('data/mixedCTEM_tomo%d',i);
    
    tomos.tomo_id(i) = i;
    
    tomos.set_stack (i,[tomo_base '.b4.ali.mrc']);
    tomos.set_angles(i,[tomo_base '.tlt']);
    
    tomos.set_defocus(i,[tomo_base '.defocus']);
    
    tomos.pix_size(i,:)  = apix;
    tomos.tomo_size(i,:) = tsiz;
end
tomos.save('tomos_b4.tomostxt');

%%

% Load BIN4 tomosinfo
tomos = SUSAN.read('tomos_b4.tomostxt');

% Load manually picked particles (in bin8)
tbl = SUSAN.read('picked_b8.tbl'); 

% Scale locations from bin8 to bin4
tbl(:,[24 25 26]) = 2*tbl(:,[24 25 26]);

% Randomize orientations
tbl(:,[7 8 9]) = 360*rand(size(tbl(:,[7 8 9])));

% Create PtclsInfo and save it.
ptcls = SUSAN.Data.ParticlesInfo(tbl,tomos);
ptcls.save('prj_001.ptclsraw');

%% Reconstruct initial PtclsInfo:

avgr = SUSAN.Modules.Averager;
avgr.gpu_list = [0 1 2 3 4 5 6];
avgr.set_ctf_correction('phase_flip');
avgr.rec_halves = false;
avgr.reconstruct('prj_001','tomos_b4.tomostxt','prj_001.ptclsraw',64);

%% Create a MRA PtclsInfo (2 classes):

ptcls = SUSAN.read('prj_001.ptclsraw');
SUSAN.Data.Particles.MRA.duplicate(ptcls,1);
ptcls.save('prj_001_mra.ptclsraw');

%% Create spherical mask and a reference for the goldbeads:

SUSAN.IO.write_mrc( -dynamo_sphere(4,64,[33 33 33],2), 'gb_b4.mrc', 2.62*4 );
SUSAN.IO.write_mrc( dynamo_sphere(21,64,[33 33 33],3), 'mask_sph_b4.mrc', 2.62*4 );

%% Decompress ribosome reference (if compressed)

if( SUSAN.Utils.exist_file('emd_3420_b4.mrc.gz') )
    system('gunzip emd_3420_b4.mrc.gz');
end

%% Create ReferenceInfo with 2 classes:

refs = SUSAN.Data.ReferenceInfo.create(2);
refs(1).map  = 'emd_3420_b4.mrc';
refs(1).mask = 'mask_sph_b4.mrc';
refs(2).map  = 'gb_b4.mrc';
refs(2).mask = 'mask_sph_b4.mrc';
refs.save(refs,'prj_001_b4.refstxt');

%%

mngr = SUSAN.Project.Manager('prj_001_b4',64);

mngr.initial_reference = 'prj_001_b4.refstxt';
mngr.initial_particles = 'prj_001_mra.ptclsraw';
mngr.tomogram_file     = 'tomos_b4.tomostxt';

mngr.gpu_list = [0 1 2 3];

mngr.aligner.set_ctf_correction('on_reference');
mngr.aligner.set_normalization('zm1s');
mngr.aligner.padding = 0;
mngr.aligner.drift = true;
mngr.averager.set_ctf_correction('phase_flip');

%% Common for all iterations:

mngr.cc_threshold = 0.9;
mngr.alignment_type = 3;
mngr.aligner.bandpass.highpass = 0;

%%

for i = 1
    mngr.aligner.set_angular_search(360,30,360,30);
    mngr.aligner.set_angular_refinement(2,1);
    mngr.aligner.set_offset_ellipsoid(24,1);
    mngr.aligner.bandpass.lowpass = 21;
    mngr.execute_iteration(i);
end

%%

for i = 2
    mngr.aligner.set_angular_search(0,1,0,1);
    mngr.aligner.set_angular_refinement(0,1);
    mngr.aligner.set_offset_ellipsoid(12,1);
    mngr.aligner.bandpass.lowpass = 21;
    mngr.execute_iteration(i);
end

%%

for i = 3:5
    mngr.aligner.set_angular_search(32,6,32,6);
    mngr.aligner.set_angular_refinement(2,1);
    mngr.aligner.set_offset_ellipsoid(20,1);
    mngr.aligner.bandpass.lowpass = 21;
    mngr.execute_iteration(i);
end

%% Visualize maps (Iteration 5, classes 1 and 2):

dtmshow( mngr.get_map(5,1) );
dtmshow( mngr.get_map(5,2) );

%% Show FSCs: Iterations from 1 to 5, class 1

figure; mngr.show_fsc( 1:5, 1 );

%% Show FSCs: Iterations from 1 to 5, class 2

figure; mngr.show_fsc( 1:5, 2 );

%% Show FSCs: Iteration 5, class 1 and 2

figure; mngr.show_fsc( 5, 1:2 );

%% Get Resulting PtclsInfo and select the first class

p_rslt   = mngr.get_ptcls(5);
p_class1 = SUSAN.Data.Particles.MRA.select_refs(p_rslt,1);

%% Show CC distribution:

figure;
hist(p_class1.ali_cc,200);

%% Update location of the particles and discard bad particles.
p_class1.update_position();
p_class1_exc = SUSAN.Data.Particles.Geom.discard_closer(p_class1,10); % Discard particles closer than 10A.
pw = p_class1_exc.select( p_class1_exc.ali_cc>0.4 );
pw.save('prj_002.ptclsraw');

%% Reconstruct BIN2:

avgr = SUSAN.Modules.Averager;
avgr.gpu_list = [0 1 2 3];
avgr.set_ctf_correction('phase_flip');
avgr.rec_halves = true;
avgr.reconstruct('prj_002','tomos_b2.tomostxt','prj_002.ptclsraw',128);

%% Create mask:

a = dread('prj_002_class001.mrc');
b = dbandpass(a,[0 20 2]);
h = dynamo_sphere(4,10);
h = h/sum(h(:));
c = convn( single(b<-1),h,'same' );
h = dynamo_sphere(3,8);
h = h/sum(h(:));
m = convn( single(c>0.02),h,'same' );
dwrite(m,'mask_b2_01.mrc');

%%

fsc = SUSAN.Utils.fsc_get('prj_002_class001_half1.mrc','prj_002_class001_half2.mrc','mask_b2_01.mrc');
figure;
plot(fsc);

%% Create ReferenceInfo with 1 classes:

refs = SUSAN.Data.ReferenceInfo.create(1);
refs(1).map  = 'prj_002_class001.mrc';
refs(1).mask = 'mask_b2_01.mrc';
refs.save(refs,'prj_002.refstxt');

%%

mngr = SUSAN.Project.Manager('prj_002_b2',128);

mngr.initial_reference = 'prj_002.refstxt';
mngr.initial_particles = 'prj_002.ptclsraw';
mngr.tomogram_file     = 'tomos_b2.tomostxt';

mngr.gpu_list = [0 1 2 3];

mngr.aligner.set_ctf_correction('cfsc');
mngr.aligner.set_normalization('zm1s');
mngr.aligner.padding = 0;
mngr.aligner.drift = true;
mngr.averager.set_ctf_correction('phase_flip');

%% Common for all iterations:

mngr.cc_threshold = 0.9;
mngr.alignment_type = 3;
mngr.aligner.bandpass.highpass = 0;

%% Closed-loop iterations (feedback)

low_pass = 25;
for i = 1:10
    as = atan2d(1,low_pass);
    mngr.aligner.set_angular_search(4*as,as,4*as,as);
    mngr.aligner.set_angular_refinement(1,2);
    mngr.aligner.set_offset_ellipsoid(10,1);
    mngr.aligner.bandpass.lowpass = low_pass;
    new_low_pass = mngr.execute_iteration(i);
    low_pass = min( low_pass+2, new_low_pass );
end

%%

figure;
mngr.show_fsc(1:3:10);

%%

low_pass = 40;
for i = 11
    as = atan2d(1,low_pass);
    mngr.aligner.set_angular_search(6*as,as,6*as,as);
    mngr.aligner.set_angular_refinement(0,1);
    mngr.aligner.set_offset_ellipsoid(12,1);
    mngr.aligner.bandpass.lowpass = low_pass;
    low_pass = mngr.execute_iteration(i);
end

%%

low_pass = 41;
for i = 12:20
    as = atan2d(1,low_pass);
    mngr.aligner.set_angular_search(4*as,as,4*as,as);
    mngr.aligner.set_angular_refinement(1,2);
    mngr.aligner.set_offset_ellipsoid(8,1);
    mngr.aligner.bandpass.lowpass = low_pass;
    new_low_pass = mngr.execute_iteration(i);
    low_pass = min( low_pass+2, new_low_pass );
end

%%

low_pass = 55;
as = atan2d(1,low_pass);
mngr.aligner.set_angular_search(6*as,as,6*as,as);
mngr.aligner.set_angular_refinement(1,1);
mngr.aligner.set_offset_ellipsoid(8,1);
mngr.aligner.bandpass.lowpass = low_pass;

for i = 21:25
    mngr.execute_iteration(i);
end

%%

p_1 = mngr.get_ptcls(25);
p_1.update_position();
p_2 = SUSAN.Data.Particles.Geom.discard_closer(p_1,12);
p_3 = p_2.select( p_2.ali_cc<0.14 );
p_3.halfsets_by_Y();
p_3.save('prj_003.ptclsraw');

%%

ctf_est = SUSAN.Modules.CtfEstimator(400);
ctf_est.gpu_list = [0 1 2 3];
ctf_est.resolution.min = 30;
ctf_est.resolution.max = 8.5;
ctf_est.defocus.min = 10000;
ctf_est.defocus.max = 50000;
ctf_est.refine_def.range = 4000;
ctf_est.refine_def.step   = 10;
ctf_est.tlt_range = 6000;
ctf_est.verbose = 2;
tomos_ctf = ctf_est.estimate('ctf_grid','prj_003.ptclsraw','tomos_b1.tomostxt');
tomos_ctf.defocus(:,7,:) = 0; % Max resolution
tomos_ctf.defocus(:,6,:) = 4*( abs( tomos_ctf.proj_eZYZ(:,2,:) )/2 ); % Exposure filter
tomos_ctf.save('tomos_b1_ctf.tomostxt');

%%

ptcls = SUSAN.read('prj_003.ptclsraw');
ptcls.update_defocus(tomos_ctf);
ptcls.save('prj_003.ptclsraw');

%%

avgr = SUSAN.Modules.Averager;
avgr.gpu_list = [0 1 2 3];
avgr.set_ctf_correction('wiener_ssnr',0.2);
avgr.rec_halves = true;
avgr.reconstruct('prj_003','tomos_b1_ctf.tomostxt','prj_003.ptclsraw',256);

%% Create mask:

a = dread('prj_003_class001.mrc');
b = dbandpass(a,[0 50 2]);
h = dynamo_sphere(8,18);
h = h/sum(h(:));
c = convn( single(b<-1),h,'same' );
h = dynamo_sphere(5,12);
h = h/sum(h(:));
m = convn( single(c>0.02),h,'same' );
dwrite(m,'mask_b1_01.mrc');

%%

fsc = SUSAN.Utils.fsc_get('prj_003_class001_half1.mrc','prj_003_class001_half2.mrc','mask_b1_01.mrc');
figure;
plot(fsc);

%% Create ReferenceInfo with 1 classes:

refs = SUSAN.Data.ReferenceInfo.create(1);
refs(1).map  = 'prj_003_class001.mrc';
refs(1).mask = 'mask_b1_01.mrc';
refs.save(refs,'prj_003.refstxt');

%%

mngr = SUSAN.Project.Manager('prj_003_b1',256);

mngr.initial_reference = 'prj_003.refstxt';
mngr.initial_particles = 'prj_003.ptclsraw';
mngr.tomogram_file     = 'tomos_b1_ctf.tomostxt';

mngr.gpu_list = [0 1 2 3];

mngr.aligner.set_ctf_correction('cfsc');
mngr.aligner.set_normalization('zm1s');
mngr.aligner.padding = 0;
mngr.aligner.drift = true;
mngr.averager.set_ctf_correction('wiener_ssnr',0.2);

%% Common for all iterations:

mngr.cc_threshold = 0.9;
mngr.alignment_type = 3;
mngr.aligner.bandpass.highpass = 0;

%% closed-loop iteration

low_pass = 36;
for i = 1:10
    as = atan2d(1,low_pass);
    mngr.aligner.set_angular_search(4*as,as,4*as,as);
    mngr.aligner.set_angular_refinement(0,1);
    mngr.aligner.set_offset_ellipsoid(10,1);
    mngr.aligner.bandpass.lowpass = low_pass;
    new_low_pass = mngr.execute_iteration(i);
    low_pass = min( low_pass+3, new_low_pass );
end

%% Using FOM filter and SSNR reconstruction to enhance reference.

mngr.averager.set_ctf_correction('wiener_ssnr',0.5,0.2);
fsc_filter = SUSAN.Utils.fsc_filter(mngr.box_size);
low_pass = 50;
for i = 11:15
    as = atan2d(1,low_pass);
    mngr.cc_threshold = 0.9;
    mngr.alignment_type = 3;
    mngr.aligner.set_angular_search(6*as,as,6*as,as);
    mngr.aligner.set_angular_refinement(0,1);
    mngr.aligner.set_offset_ellipsoid(10,1);
    mngr.aligner.bandpass.highpass = 0;
    mngr.aligner.bandpass.lowpass = low_pass;
    new_low_pass = mngr.execute_iteration(i);
    low_pass = min( low_pass+2, new_low_pass );
    
    fsc_filter.set_fsc_FOM(mngr.get_fsc(i));
    
    m = mngr.get_name_map(i);    
    [v,apix] = SUSAN.IO.read_mrc(m);
    v_enh = fsc_filter.apply_filter(v);
    SUSAN.IO.write_mrc(v_enh,m,apix);
end

%% Using FOM filter and SSNR reconstruction to enhance reference.

mngr.averager.set_ctf_correction('wiener_ssnr',0.5,0.2);
fsc_filter = SUSAN.Utils.fsc_filter(mngr.box_size);
low_pass = 60;
for i = 16:20
    as = atan2d(1,low_pass);
    mngr.cc_threshold = 0.9;
    mngr.alignment_type = 3;
    mngr.aligner.set_angular_search(4*as,as,4*as,as);
    mngr.aligner.set_angular_refinement(1,2);
    mngr.aligner.set_offset_ellipsoid(10,1);
    mngr.aligner.bandpass.highpass = 0;
    mngr.aligner.bandpass.lowpass = low_pass;
    new_low_pass = mngr.execute_iteration(i);
    low_pass = min( low_pass+2, new_low_pass );
    
    fsc_filter.set_fsc_FOM(mngr.get_fsc(i));
    
    m = mngr.get_name_map(i);    
    [v,apix] = SUSAN.IO.read_mrc(m);
    v_enh = fsc_filter.apply_filter(v);
    SUSAN.IO.write_mrc(v_enh,m,apix);
end

%%

p_1 = mngr.get_ptcls(20);
p_1.update_position();
p_2 = SUSAN.Data.Particles.Geom.discard_closer(p_1,12);
p_2.halfsets_by_Y();
p_2.class_cix = rand(size(p_2.class_cix))<0.8;
SUSAN.Data.Particles.MRA.duplicate(p_2,1);
p_2.save('prj_004.ptclsraw');

%%

avgr = SUSAN.Modules.Averager;
avgr.gpu_list = [0 1 2 3];
avgr.set_ctf_correction('wiener_ssnr',0.5,0.2);
avgr.rec_halves = true;
avgr.reconstruct('prj_004','tomos_b1_ctf.tomostxt','prj_004.ptclsraw',256);

%%

refs = SUSAN.Data.ReferenceInfo.create(2);
refs(1).map  = 'prj_004_class001.mrc';
refs(1).mask = 'mask_b1_01.mrc';
refs(2).map  = 'prj_004_class002.mrc';
refs(2).mask = 'mask_b1_01.mrc';
refs.save(refs,'prj_004.refstxt');

%%

mngr = SUSAN.Project.Manager('prj_004_b1',256);

mngr.initial_reference = 'prj_004.refstxt';
mngr.initial_particles = 'prj_004.ptclsraw';
mngr.tomogram_file     = 'tomos_b1_ctf.tomostxt';

mngr.gpu_list = [0 1 2 3];

mngr.aligner.set_ctf_correction('cfsc');
mngr.aligner.set_normalization('zm1s');
mngr.aligner.padding = 0;
mngr.aligner.drift = true;
mngr.averager.set_ctf_correction('wiener_ssnr',0.5,0.2);

%% Common for all iterations:

mngr.cc_threshold = 0.9;
mngr.alignment_type = 3;
mngr.aligner.bandpass.highpass = 0;

%%

low_pass = 65;
for i = 1:8
    as = atan2d(1,low_pass);
    mngr.aligner.set_angular_search(2*as,as,2*as,as);
    mngr.aligner.set_angular_refinement(0,1);
    mngr.aligner.set_offset_ellipsoid(20,1);
    mngr.aligner.bandpass.lowpass = low_pass;
    mngr.execute_iteration(i);
end

%%

p_rslt   = mngr.get_ptcls(8);
p_class1 = SUSAN.Data.Particles.MRA.select_refs(p_rslt,2);
p_class1_exc = SUSAN.Data.Particles.Geom.discard_closer(p_class1,12);
p_class1_exc.save('prj_005.ptclsraw');

%%

avgr = SUSAN.Modules.Averager;
avgr.gpu_list = [0 1 2 3];
avgr.set_ctf_correction('wiener_ssnr',0.5,0.3);
avgr.rec_halves = true;
avgr.reconstruct('prj_005','tomos_b1_ctf.tomostxt','prj_005.ptclsraw',256);

%%

refs = SUSAN.Data.ReferenceInfo.create(1);
refs(1).map  = 'prj_005_class001.mrc';
refs(1).mask = 'mask_b1_01.mrc';
refs.save(refs,'prj_005.refstxt');

%%

mngr = SUSAN.Project.Manager('prj_005_b1',256);

mngr.initial_reference = 'prj_005.refstxt';
mngr.initial_particles = 'prj_005.ptclsraw';
mngr.tomogram_file     = 'tomos_b1_ctf.tomostxt';

mngr.gpu_list = [0 1 2 3];

mngr.aligner.set_ctf_correction('cfsc');
mngr.aligner.set_normalization('zm1s');
mngr.aligner.padding = 0;
mngr.aligner.drift = true;
mngr.averager.set_ctf_correction('wiener_ssnr',0.5,0.25);

%% Common for all iterations:

mngr.cc_threshold = 1;
mngr.aligner.bandpass.highpass = 0;

%% closed-loop iteration

mngr.alignment_type = 3;
low_pass = 60;
for i = 1:5
    as = atan2d(1,low_pass);
    mngr.aligner.set_angular_search(4*as,as,4*as,as);
    mngr.aligner.set_angular_refinement(1,1);
    mngr.aligner.set_offset_ellipsoid(10,1);
    mngr.aligner.bandpass.lowpass = low_pass;
    new_low_pass = mngr.execute_iteration(i);
    low_pass = min( low_pass+2, new_low_pass );
end

%%

mngr.aligner.halfsets = 1;
mngr.alignment_type = 2;
low_pass = 60;
for i = 6:10
    as = atan2d(1,low_pass);
    mngr.aligner.set_angular_search(4*as,as,4*as,as);
    mngr.aligner.set_angular_refinement(1,1);
    mngr.aligner.set_offset_ellipsoid(10,1);
    mngr.aligner.bandpass.lowpass = low_pass;
    new_low_pass = mngr.execute_iteration(i);
    low_pass = min( low_pass+2, new_low_pass );
end

%%

p_1 = mngr.get_ptcls(10);

%% Show CC distribution per projection:

figure;
hist(p_1.prj_cc(p_1.prj_w>0),200);

%% calculate dose weight by CC:

p_1.prj_w( p_1.prj_cc > 0.01 ) = 0;
p_1.prj_w( p_1.prj_cc < 0.004 ) = 0;
cc = p_1.prj_cc;
cc(p_1.prj_w<1e-6) = 0;
cc = cc-min(cc(cc>0));
cc = 1-cc/max(cc(:));
cc( cc>1 ) = 0;
p_1.defocus(:,6,:) = 150*cc + 20;
p_1.save('prj_006.ptclsraw');

%%

avgr = SUSAN.Modules.Averager;
avgr.gpu_list = [0 1 2 3];
avgr.set_ctf_correction('wiener_ssnr',0.6,0.3);
avgr.rec_halves = true;
avgr.reconstruct('prj_006','tomos_b1_ctf.tomostxt','prj_006.ptclsraw',256);




















