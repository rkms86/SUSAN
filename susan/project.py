###########################################################################
# This file is part of the Substack Analysis (SUSAN) framework.
# Copyright (c) 2018-2021 Ricardo Miguel Sanchez Loayza.
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
###########################################################################

import numpy as _np

import susan.data    as _ssa_data
import susan.utils   as _ssa_utils
import susan.modules as _ssa_modules

from susan.io.mrc import read     as _mrc_read
from susan.io.mrc import get_info as _mrc_get_info

import susan.utils.datatypes  as _dt
import susan.utils.txt_parser as _prsr

from os      import remove as _rm
from os      import mkdir  as _mkdir
from os.path import exists as _file_exists

class _iteration_files:
    def __init__(self):
        self.ptcl_rslt = ''
        self.ptcl_temp = ''
        self.reference = ''
        self.ite_dir   = ''
        
    def check(self):
        if not _file_exists(self.ptcl_rslt):
            raise NameError('File '+ self.ptcl_rslt + ' does not exist')
        if not _file_exists(self.reference):
            raise NameError('File '+ self.reference + ' does not exist')

class Manager:
    def __init__(self,prj_name,box_size=None):
        if box_size is None:
            fp = open(prj_name+"/info.prjtxt","r")
            self.prj_name = _prsr.read(fp,'name')
            self.box_size = int(_prsr.read(fp,'box_size'))
            fp.close()
        else:
            if not _file_exists(prj_name):
                _mkdir(prj_name)
            fp = open(prj_name+"/info.prjtxt","w")
            _prsr.write(fp,'name',prj_name)
            _prsr.write(fp,'box_size',str(box_size))
            fp.close()
            self.prj_name = prj_name
            self.box_size = box_size
        
        self.tomogram_file     = ''
        self.initial_reference = ''
        self.initial_particles = ''
        
        self.list_gpus_ids     = [0]
        self.threads_per_gpu   = 1
        self.iteration_type    = 3
        
        self.cc_threshold      = 0.8
        self.fsc_threshold     = 0.143
        
        self.mpi               = _dt.mpi_params('srun -n %d ',1)
        self.verbosity         = 0
        
        self.aligner           = _ssa_modules.Aligner()
        self.averager          = _ssa_modules.Averager()
        
        self.aligner.ctf_correction = 'cfsc'
        self.max_2d_delta_angs = 0
        
        self.averager.ctf_correction    = 'wiener'
        self.averager.rec_halfsets      = True
        self.averager.bandpass.highpass = 0
        self.averager.bandpass.lowpass  = (self.box_size)/2-1
    
    def get_iteration_dir(self,ite):
        return self.prj_name + '/ite_%04d/' % ite
    
    def get_iteration_files(self,ite):
        rslt = _iteration_files()
        if ite < 1:
            rslt.ptcl_rslt = self.initial_particles
            rslt.reference = self.initial_reference
        else:
            base_dir = self.get_iteration_dir(ite)
            rslt.ptcl_rslt = base_dir + 'particles.ptclsraw'
            rslt.ptcl_temp = base_dir + 'temp.ptclsraw'
            rslt.reference = base_dir + 'reference.refstxt'
            rslt.ite_dir   = base_dir
        return rslt

    def get_names_map(self,ite,ref=1):
        if ite == 0:
            refs_info = _ssa_data.Reference(self.initial_reference)
            map_name = refs_info.ref[ref-1]
        else:
            ite_dir = self.get_iteration_dir(ite)
            map_name = ite_dir + 'map_class%03d.mrc' % ref
        return map_name

    def get_names_mask(self,ite,ref=1):
        if ite == 0:
            refs_info = _ssa_data.Reference(self.initial_reference)
            mask_name = refs_info.ref[ref-1]
        else:
            refs_info = _ssa_data.Reference(self.get_iteration_dir(ite)+'reference.refstxt')
            mask_name = refs_info.ref[ref-1]
        return mask_name

    def get_names_halfmaps(self,ite,ref=1):
        if ite == 0:
            refs_info = _ssa_data.Reference(self.initial_reference)
            h1_name = refs_info.h1[ref-1]
            h2_name = refs_info.h2[ref-1]
        else:
            ite_dir = self.get_iteration_dir(ite)
            h1_name = ite_dir + 'map_class%03d_half1.mrc' % ref
            h2_name = ite_dir + 'map_class%03d_half2.mrc' % ref
        return (h1_name,h2_name)

    def get_name_refstxt(self,ite):
        files = self.get_iteration_files(ite)
        return files.reference

    def get_name_ptcls(self,ite):
        files = self.get_iteration_files(ite)
        return files.ptcl_rslt

    def get_map(self,ite,ref=1):
        v,_ = _mrc_read(self.get_names_map(ite,ref))
        return v

    def get_ptcls(self,ite):
        files = self.get_iteration_files(ite)
        return _ssa_data.Particles(files.ptcl_rslt)

    def get_cc(self,ite,ref=1):
        ptcls = self.get_ptcls(ite)
        return ptcls.ali_cc[ref-1]

    def get_fsc(self,ite,ref=1):
        i = ref-1
        refs = _ssa_data.Reference(self.get_name_refstxt(ite))
        return _ssa_utils.fsc_get(refs.h1[i],refs.h2[i],refs.msk[i])

    def setup_iteration(self,ite):
        base_dir = self.get_iteration_dir(ite)
        if not _file_exists(base_dir):
            _mkdir(base_dir)
        cur = self.get_iteration_files(ite)
        prv = self.get_iteration_files(ite-1)
        prv.check()
        return (cur,prv)
    
    def _validate_ite_type(self):
        if self.iteration_type in (3,'3','3D','3d'):
            return 3
        elif self.iteration_type in (2,'2','2D','2d'):
            return 2
        elif self.iteration_type in ('ctf','CTF','Ctf'):
            return 'ctf'
        else:
            raise ValueError('Invalid Iteration Type (accepted valud: 3, 2, "ctf")')
    
    def _exec_alignment(self,cur,prv,ite_type):
        self.aligner.list_gpus_ids     = self.list_gpus_ids
        self.aligner.threads_per_gpu   = self.threads_per_gpu
        self.aligner.dimensionality    = ite_type
        self.aligner.verbosity         = self.verbosity
        
        print( '  [%dD Alignment] Start:' % ite_type )
        
        start_time = _ssa_utils.time_now()
        if self.mpi.arg > 1:
            self.aligner.mpi.cmd = self.mpi.cmd
            self.aligner.mpi.arg = self.mpi.arg
            self.aligner.align_mpi(cur.ptcl_rslt,prv.reference,self.tomogram_file,prv.ptcl_rslt,self.box_size)
        else:
            self.aligner.align(cur.ptcl_rslt,prv.reference,self.tomogram_file,prv.ptcl_rslt,self.box_size)
        elapsed = _ssa_utils.time_now()-start_time
        
        print( '  [%dD Alignment] Finished. Elapsed time: %.1f seconds (%s).' % (ite_type,elapsed.total_seconds(),str(elapsed)) )

    def exec_estimation(self,cur,prv):
        ite_type  = self._validate_ite_type()
        
        if ite_type == 'ctf':
            raise ValueError('CTF iteration not implemented yet')
        else:
            self._exec_alignment(cur,prv,ite_type)
        
    def exec_particle_selection(self,cur,prv):
        print('  [Aligned partices] Processing:')
        ptcls_in = _ssa_data.Particles(cur.ptcl_rslt)
        
        # Limit 2D shifts:
        if self._validate_ite_type() == 2 and self.max_2d_delta_angs > 0:
            if self.aligner.allow_drift:
                print('    Limiting 2D drift to %.2f Å.' % self.max_2d_delta_angs )
                ptcls_old  = _ssa_data.Particles(prv.ptcl_rslt)
                delta_angs = ptcls_in.prj_t - ptcls_old.prj_t
                norm_angs  = _np.linalg.norm( delta_angs, axis=2 )
                scale_lim  = self.max_2d_delta_angs/_np.maximum(norm_angs,1)
                scale_lim[ norm_angs<self.max_2d_delta_angs ] = 1
                scale_lim = scale_lim[:,:,_np.newaxis]
                delta_angs = scale_lim*delta_angs
                ptcls_in.prj_t[:] = ptcls_old.prj_t + delta_angs
                ptcls_in.save(cur.ptcl_rslt)
            else:
                print('    Limiting 2D shift to %.2f Å.' % self.max_2d_delta_angs )
                norm_angs  = _np.linalg.norm( ptcls_in.prj_t, axis=2 )
                scale_lim  = self.max_2d_delta_angs/_np.maximum(norm_angs,1)
                scale_lim[ norm_angs<self.max_2d_delta_angs ] = 1
                scale_lim = scale_lim[:,:,_np.newaxis]
                ptcls_in.prj_t[:] = scale_lim*ptcls_in.prj_t
                ptcls_in.save(cur.ptcl_rslt)
        
        # Classify
        if ptcls_in.n_refs > 1 :
            ptcls_in.ref_cix = _np.argmax(ptcls_in.ali_cc,axis=0)
            ptcls_in.save(cur.ptcl_rslt)
        
        # Select particles for reconstruction
        for i in range(ptcls_in.n_refs):
            idx = (ptcls_in.ref_cix == i).flatten()
            if _np.any( idx ):
                hid = ptcls_in.half_id[idx].flatten()
                ccc = ptcls_in.ali_cc [i,idx].flatten()

                n_rf = hid.shape[0]
                n_h1 = (hid==1).sum()
                n_h2 = (hid==2).sum()
                
                if n_h1 > 0:
                    th1 = _np.quantile(ccc[ hid==1 ], 1-self.cc_threshold)
                    hid[ (hid==1) & (ccc<th1) ] = 0

                if n_h2 > 0:
                    th2 = _np.quantile(ccc[ hid==2 ], 1-self.cc_threshold)
                    hid[ (hid==2) & (ccc<th2) ] = 0
            
                ptcls_in.half_id[idx] = hid
            
                print('    Class %2d: %7d particles [%7d].' % (i+1,n_rf,(hid >0).sum()) )
                print('      Half 1: %7d particles [%7d].'  % (    n_h1,(hid==1).sum()) )
                print('      Half 2: %7d particles [%7d].'  % (    n_h2,(hid==2).sum()) )
            else:
                print('    Class %2d: %7d particles.' % (i+1,0) )
                print('      Half 1: %7d particles.'  % ( 0 ) )
                print('      Half 2: %7d particles.'  % ( 0 ) )

            
        ptcls_out = ptcls_in[ (ptcls_in.half_id>0).flatten() ]
        ptcls_out.save(cur.ptcl_temp)
        print('  [Aligned partices] Done.')
        
    def exec_averaging(self,cur,prv):
        self.averager.list_gpus_ids     = self.list_gpus_ids
        self.averager.threads_per_gpu   = self.threads_per_gpu
        self.averager.verbosity         = self.verbosity
        
        print( '  [Reconstruct Maps] Start:' )
        start_time = _ssa_utils.time_now()
        if self.mpi.arg > 1:
            self.averager.mpi.cmd = self.mpi.cmd
            self.averager.mpi.arg = self.mpi.arg
            self.averager.reconstruct_mpi(cur.ite_dir+'map',self.tomogram_file,cur.ptcl_temp,self.box_size)
        else:
            self.averager.reconstruct(cur.ite_dir+'map',self.tomogram_file,cur.ptcl_temp,self.box_size)
        elapsed = _ssa_utils.time_now()-start_time
        
        print( '  [Reconstruct Maps] Finished. Elapsed time: %.1f seconds (%s).' % (elapsed.total_seconds(),str(elapsed)) )
        
        _rm(cur.ptcl_temp)
        
        refs = _ssa_data.Reference(prv.reference)
        for i in range(refs.n_refs):
            refs.ref[i] = '%s/map_class%03d.mrc'       % (cur.ite_dir,i+1)
            refs.h1[i]  = '%s/map_class%03d_half1.mrc' % (cur.ite_dir,i+1)
            refs.h2[i]  = '%s/map_class%03d_half2.mrc' % (cur.ite_dir,i+1)
        refs.save(cur.reference)
    
    def exec_postprocessing(self,cur):
        refs = _ssa_data.Reference(cur.reference)
        if refs.n_refs == 1:
            print( '  [FSC Calculation] Start (1 reference):' )
        else:
            print( '  [FSC Calculation] Start (%d references):' % refs.n_refs )
        
        rslt = _np.zeros( (refs.n_refs) )
        for i in range(refs.n_refs):
            fsc = _ssa_utils.fsc_get(refs.h1[i],refs.h2[i],refs.msk[i])
            _,pix_size,_ = _mrc_get_info(refs.ref[i])
            fsc_rslt = _ssa_utils.fsc_analyse(fsc,pix_size,self.fsc_threshold)
            print('    - Reference %2d: %7.3f angstroms [%d fourier pixels]' % (i+1,fsc_rslt.res,fsc_rslt.fpix))
            rslt[i] = fsc_rslt.fpix
        
        if refs.n_refs == 1:
            return rslt[0]
        else:
            return rslt
    
    def execute_iteration(self,ite):
        start_time = _ssa_utils.time_now()
        print('============================')
        print('Project: %s (Iteration %d)'%(self.prj_name,ite))
        cur,prv = self.setup_iteration(ite)
        self.exec_estimation(cur,prv)
        self.exec_particle_selection(cur,prv)
        self.exec_averaging(cur,prv)
        rslt = self.exec_postprocessing(cur)
        elapsed = _ssa_utils.time_now()-start_time
        print('Iteration %d Finished [Elapsed time: %.1f seconds (%s]'%(ite,elapsed.total_seconds(),str(elapsed)))
        return rslt

