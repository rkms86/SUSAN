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

import susan.utils.txt_parser as _prsr
import numpy as _np
from susan.io.mrc import get_info as _mrc_info
from susan.utils import is_extension as _is_ext
from susan.utils import force_extension as _force_ext

class Tomograms:
    
    def __init__(self,filename=None,n_ptcl=0,n_proj=0):
        if isinstance(filename, str):
            self.load(filename) 
        else:
            if n_ptcl > 0 and n_proj > 0:
                self.alloc(n_ptcl,n_proj)
            else:
                raise NameError('Invalid input')
    
    def get_n_tomos(self): return self.tomo_id.shape[0]
    def get_n_projs(self): return self.proj_eZYZ.shape[1]
    
    n_tomos = property(get_n_tomos)
    n_projs = property(get_n_projs)
    
    @staticmethod
    def _check_filename(filename):
        if not _is_ext(filename,'tomostxt'):
            raise ValueError( 'Wrong file extension, do you mean ' + _force_ext(filename,'tomostxt') + '?')
    
    #def __repr__(self):
    #    return "Tomograms"
    
    def alloc(self,n_tomos,n_projs):
        self.tomo_id    = _np.zeros( n_tomos   ,dtype=_np.uint32 )
        self.tomo_size  = _np.zeros((n_tomos,3),dtype=_np.uint32 )
        self.stack_file = []
        self.stack_size = _np.zeros((n_tomos,3),dtype=_np.uint32 )
        self.num_proj   = _np.zeros( n_tomos   ,dtype=_np.uint32 )
        self.pix_size   = _np.zeros( n_tomos   ,dtype=_np.float32)
        self.proj_eZYZ  = _np.zeros((n_tomos,n_projs,3),dtype=_np.float32)
        self.proj_shift = _np.zeros((n_tomos,n_projs,2),dtype=_np.float32)
        self.proj_wgt   = _np.zeros((n_tomos,n_projs)  ,dtype=_np.float32)
        self.voltage    = _np.zeros( n_tomos   ,dtype=_np.float32)
        self.sph_aber   = _np.zeros( n_tomos   ,dtype=_np.float32)
        self.amp_cont   = _np.zeros( n_tomos   ,dtype=_np.float32)
        
        # Defocus
        self.def_U    = _np.zeros((n_tomos,n_projs),dtype=_np.float32) # U (angstroms)
        self.def_V    = _np.zeros((n_tomos,n_projs),dtype=_np.float32) # V (angstroms)
        self.def_ang  = _np.zeros((n_tomos,n_projs),dtype=_np.float32) # angles (sexagesimal)
        self.def_phas = _np.zeros((n_tomos,n_projs),dtype=_np.float32) # phase shift (sexagesimal?)
        self.def_Bfct = _np.zeros((n_tomos,n_projs),dtype=_np.float32) # Bfactor
        self.def_ExFl = _np.zeros((n_tomos,n_projs),dtype=_np.float32) # Exposure filter
        self.def_mres = _np.zeros((n_tomos,n_projs),dtype=_np.float32) # Max. resolution (angstroms)
        self.def_scor = _np.zeros((n_tomos,n_projs),dtype=_np.float32) # score
        
        for i in range(n_tomos):
            self.stack_file.append('')

    def load(self,filename):
        Tomograms._check_filename(filename)
        
        fp = open(filename,"rb")
        n_tomos = int(_prsr.read(fp,'num_tomos'))
        n_projs = int(_prsr.read(fp,'num_projs'))
        self.alloc(n_tomos,n_projs)
        for i in range(n_tomos):
            self.tomo_id[i]      = _np.uint32((_prsr.read(fp,'tomo_id')))
            self.tomo_size[i,:]  = _np.fromstring(_prsr.read(fp,'tomo_size'),_np.uint32,sep=',')
            self.stack_file[i]   = _prsr.read(fp,'stack_file')
            self.stack_size[i,:] = _np.fromstring(_prsr.read(fp,'stack_size'),_np.uint32,sep=',')
            self.pix_size[i]     = _np.float32((_prsr.read(fp,'pix_size')))
            self.voltage[i]      = _np.float32((_prsr.read(fp,'kv')))
            self.sph_aber[i]     = _np.float32((_prsr.read(fp,'cs')))
            self.amp_cont[i]     = _np.float32((_prsr.read(fp,'ac')))
            self.num_proj[i]     = _np.uint32((_prsr.read(fp,'num_proj')))
            
            P = self.num_proj[i]
            buffer = _np.loadtxt(fp,dtype=_np.float32,comments='#',ndmin=2,max_rows=P)
            self.proj_eZYZ [i,:P,:] = buffer[:,0:3]
            self.proj_shift[i,:P,:] = buffer[:,3:5]
            self.proj_wgt  [i,:P]   = buffer[:,5]
            self.def_U     [i,:P]   = buffer[:,6]
            self.def_V     [i,:P]   = buffer[:,7]
            self.def_ang   [i,:P]   = buffer[:,8]
            self.def_phas  [i,:P]   = buffer[:,9]
            self.def_Bfct  [i,:P]   = buffer[:,10]
            self.def_ExFl  [i,:P]   = buffer[:,11]
            self.def_mres  [i,:P]   = buffer[:,12]
            self.def_scor  [i,:P]   = buffer[:,13]
    
    def save(self,filename):
        Tomograms._check_filename(filename)
        
        fp=open(filename,'w')
        _prsr.write(fp,'num_tomos',str(self.n_tomos))
        _prsr.write(fp,'num_projs',str(self.n_projs))
        for i in range(self.n_tomos):
            fp.write('## Tomogram/Stack '+str(i+1)+'\n')
            _prsr.write(fp,'tomo_id'   , str(self.tomo_id[i]))
            _prsr.write(fp,'tomo_size' , '%d,%d,%d'%(self.tomo_size[i,0],self.tomo_size[i,1],self.tomo_size[i,2]))
            _prsr.write(fp,'stack_file', str(self.stack_file[i] ))
            _prsr.write(fp,'stack_size','%d,%d,%d'%(self.stack_size[i,0],self.stack_size[i,1],self.stack_size[i,2]))
            _prsr.write(fp,'pix_size'  , str(self.pix_size[i]))
            _prsr.write(fp,'kv'        , str(self.voltage[i]))
            _prsr.write(fp,'cs'        , str(self.sph_aber[i]))
            _prsr.write(fp,'ac'        , str(self.amp_cont[i]))
            _prsr.write(fp,'num_proj'  , str(self.num_proj[i]))
            
            fp.write('#euler.Z  euler.Y  euler.Z  shift.X  shift.Y    weight')
            fp.write('  Defocus.U  Defocus.V  Def.ang  PhShift')
            fp.write('  BFactor  ExpFilt')
            fp.write(' Res.angs FitScore\n')
            
            P = self.num_proj[i]
            for j in range(P):
                fp.write('%8.3f %8.3f %8.3f ' % (self.proj_eZYZ[i,j,0],self.proj_eZYZ[i,j,1],self.proj_eZYZ[i,j,2]))
                fp.write('%8.2f %8.2f '       % (self.proj_shift[i,j,0],self.proj_shift[i,j,1]))
                fp.write('%9.4f '             % (self.proj_wgt[i,j]))
                fp.write('%10.2f %10.2f '     % (self.def_U   [i,j],self.def_V   [i,j])) # Defocus.U    Defocus.V
                fp.write('%8.3f %8.3f '       % (self.def_ang [i,j],self.def_phas[i,j])) # Def.ang      Def.ph_shft
                fp.write('%8.2f %8.2f '       % (self.def_Bfct[i,j],self.def_ExFl[i,j])) # Def.BFactor  Def.ExpFilt
                fp.write('%8.4f %8.5f '       % (self.def_mres[i,j],self.def_scor[i,j])) # Def.max_res  Def.score
                fp.write('\n')
        fp.close()
    
    def set_stack(self,idx,stk_name):
        stk_dims,apix,_ = _mrc_info(stk_name)
        P = stk_dims[2]
        self.stack_file[idx]   = stk_name
        self.stack_size[idx,:] = stk_dims
        self.pix_size[idx]     = apix[:2].mean()
        self.num_proj[idx]     = P
        self.proj_wgt[idx,:]   = 0
        self.proj_wgt[idx,:P]  = 1
    
    def set_angles(self,idx,tlt_file):
        angs = _np.loadtxt(tlt_file)
        self.proj_eZYZ[idx,:,:] = 0
        self.proj_eZYZ[idx,:self.num_proj[idx],1] = angs

    def set_defocus(self,idx,def_file):
        if( _is_ext(def_file,'defocus') ):
            line = _np.loadtxt(def_file,dtype=_np.float32,comments='#',max_rows=1)
            version = int(line[-1])
            if version == 2:
                fp=open(def_file,'r')
                for i in range(self.num_proj[i]):
                    line = _np.loadtxt(fp,dtype=_np.float32,comments='#',max_rows=1)
                    self.def_U  [idx,i] = 10*line[4]
                    self.def_V  [idx,i] = 10*line[4]
                    self.def_ang[idx,i] = 0
                fp.close()
            elif version == 3:
                fp=open(def_file,'r')
                for i in range(self.num_proj[i]):
                    line = _np.loadtxt(fp,dtype=_np.float32,comments='#',max_rows=1)
                    self.def_U  [idx,i] = 10*line[4]
                    self.def_V  [idx,i] = 10*line[5]
                    self.def_ang[idx,i] = line[6]
                fp.close()
            else:
                raise NameError('Invalid DEFOCUS format')
        elif _is_ext(def_file,'txt'):
            P = self.num_proj[i]
            buffer = _np.loadtxt(def_file,dtype=_np.float32,comments='#',ndmin=2,max_rows=P)
            self.def_U     [idx,:P]   = buffer[:,0]
            self.def_V     [idx,:P]   = buffer[:,1]
            self.def_ang   [idx,:P]   = buffer[:,2]
            self.def_phas  [idx,:P]   = buffer[:,3]
            self.def_Bfct  [idx,:P]   = buffer[:,4]
            self.def_ExFl  [idx,:P]   = buffer[:,5]
            self.def_mres  [idx,:P]   = buffer[:,6]
            self.def_scor  [idx,:P]   = buffer[:,7]
        else:
            raise NameError('Invalid filename')
