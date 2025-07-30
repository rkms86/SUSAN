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
from susan.io import tlt as _tlt
from susan.io import xf  as _xf

from susan.utils import is_extension as _is_ext
from susan.utils import force_extension as _force_ext 
from susan.utils import euZYZ_rotm as _euZYZ_rotm
from susan.utils import rotm_euZYZ as _rotm_euZYZ

class Tomograms:
    
    def __init__(self,filename=None,n_tomo=0,n_proj=0):
        if isinstance(filename, str):
            self._load(filename) 
        else:
            if n_tomo > 0 and n_proj > 0:
                self._alloc(n_tomo,n_proj)
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
    
    def _alloc(self,n_tomos,n_projs):
        self.tomo_id    = _np.zeros( n_tomos   ,dtype=_np.uint32 )
        self.tomo_size  = _np.zeros((n_tomos,3),dtype=_np.uint32 )
        self.stack_file = []
        self.stack_size = _np.zeros((n_tomos,3),dtype=_np.uint32 )
        self.num_proj   = _np.zeros( n_tomos   ,dtype=_np.uint32 )
        self.pix_size   = _np.zeros( n_tomos   ,dtype=_np.float32)
        self.proj_eZYZ  = _np.zeros((n_tomos,n_projs,3),dtype=_np.float32)
        self.proj_shift = _np.zeros((n_tomos,n_projs,2),dtype=_np.float32)
        self.proj_wgt   = _np.zeros((n_tomos,n_projs)  ,dtype=_np.float32)
        self.voltage    = 300 *_np.ones( n_tomos,dtype=_np.float32)
        self.sph_aber   = 2.7 *_np.ones( n_tomos,dtype=_np.float32)
        self.amp_cont   = 0.07*_np.ones( n_tomos,dtype=_np.float32)
        
        # Defocus
        self.def_U    = _np.zeros((n_tomos,n_projs),dtype=_np.float32) # U (angstroms)
        self.def_V    = _np.zeros((n_tomos,n_projs),dtype=_np.float32) # V (angstroms)
        self.def_ang  = _np.zeros((n_tomos,n_projs),dtype=_np.float32) # angles (sexagesimal)
        self.def_phas = _np.zeros((n_tomos,n_projs),dtype=_np.float32) # phase shift (sexagesimal?)
        self.def_Bfct = _np.zeros((n_tomos,n_projs),dtype=_np.float32) # Bfactor
        self.def_ExFl = _np.zeros((n_tomos,n_projs),dtype=_np.float32) # Exposure filter
        self.def_mres = _np.zeros((n_tomos,n_projs),dtype=_np.float32) # Max. resolution (angstroms)
        self.def_scor = _np.zeros((n_tomos,n_projs),dtype=_np.float32) # score
        
        # Doses
        self.doses = _np.zeros((n_tomos,n_projs),dtype=_np.float32)
        
        # Nominal tilt angles (sorting reasons)
        self.nominal_tilt_angles = _np.zeros((n_tomos,n_projs),dtype=_np.float32)
	
        for i in range(n_tomos):
            self.stack_file.append('')

    def _load(self,filename):
        Tomograms._check_filename(filename)
        
        fp = open(filename,"rb")
        n_tomos = int(_prsr.read(fp,'num_tomos'))
        n_projs = int(_prsr.read(fp,'num_projs'))
        self._alloc(n_tomos,n_projs)
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
            for p in range(P):
                buffer = _np.fromstring(_prsr.read_line(fp),dtype=_np.float32,sep=' ')
                self.proj_eZYZ [i,p,:] = buffer[0:3]
                self.proj_shift[i,p,:] = buffer[3:5]
                self.proj_wgt  [i,p]   = buffer[5]
                self.def_U     [i,p]   = buffer[6]
                self.def_V     [i,p]   = buffer[7]
                self.def_ang   [i,p]   = buffer[8]
                self.def_phas  [i,p]   = buffer[9]
                self.def_Bfct  [i,p]   = buffer[10]
                self.def_ExFl  [i,p]   = buffer[11]
                self.def_mres  [i,p]   = buffer[12]
                self.def_scor  [i,p]   = buffer[13]
                if len(buffer) > 14:
                    self.doses[i,p]    = buffer[14]
                if len(buffer) > 15:
              	    self.nominal_tilt_angles[i,p] = buffer[15]
      
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
            fp.write(' Res.angs FitScore')
            fp.write(' Doses NominalTiltAngles\n')
            
            P = self.num_proj[i]
            for p in range(P):
                fp.write('%8.3f %8.3f %8.3f ' % (self.proj_eZYZ[i,p,0],self.proj_eZYZ[i,p,1],self.proj_eZYZ[i,p,2]))
                fp.write('%8.2f %8.2f '       % (self.proj_shift[i,p,0],self.proj_shift[i,p,1]))
                fp.write('%9.4f '             % (self.proj_wgt[i,p]))
                fp.write('%10.2f %10.2f '     % (self.def_U   [i,p],self.def_V   [i,p]))             # Defocus.U    Defocus.V
                fp.write('%8.3f %8.3f '       % (self.def_ang [i,p],self.def_phas[i,p]))             # Def.ang      Def.ph_shft
                fp.write('%8.2f %8.2f '       % (self.def_Bfct[i,p],self.def_ExFl[i,p]))             # Def.BFactor  Def.ExpFilt
                fp.write('%8.4f %8.5f '       % (self.def_mres[i,p],self.def_scor[i,p]))             # Def.max_res  Def.score
                fp.write('%8.4f %8.4f '       % (self.doses[i,p]   ,self.nominal_tilt_angles[i,p]))  # Dose         NominalTiltAngle
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
    
    def set_angles(self, idx, tlt_filename, xf_filename = None, xf_apix = None):
        self.proj_eZYZ [idx,:,:] = 0
        self.proj_shift[idx,:,:] = 0
        self.proj_wgt  [idx,:, ] = 0
        
        tlt = _tlt.read(tlt_filename)
        self.nominal_tilt_angles[idx, :self.num_proj[idx]] = tlt
        if (xf_filename == None):
            self.proj_eZYZ[idx, :self.num_proj[idx], 1] = tlt
            self.proj_wgt [idx, :self.num_proj[idx],  ] = 1
        else:
            xf = _xf.read(xf_filename)
            if (xf_apix != None):
                apix = xf_apix
            else:
                apix = self.pix_size[idx]
            for i in range(self.num_proj[idx]):
                rot_tlt = _np.zeros([3,3])
                _euZYZ_rotm(rot_tlt, _np.array([0.0, tlt[i], 0.0]) * _np.pi / 180.0)
                rot_xf  = _np.array([[xf[i,0,0], xf[i,0,1], 0],
                                     [xf[i,1,0], xf[i,1,1], 0],
                                     [0        , 0        , 1],
                                    ]).T
                
                vec_xf   = _np.array([xf[i,0,2], xf[i,1,2], 0]) * apix
                vec      = -rot_xf @ vec_xf
                rot      =  rot_xf @ rot_tlt
                euler_xf = _np.zeros(3)
                _rotm_euZYZ(euler_xf, rot_xf)
                self.proj_eZYZ [idx, i, 0] = euler_xf[-1] * 180.0 / _np.pi
                self.proj_eZYZ [idx, i, 1] = tlt[i]
                self.proj_shift[idx, i, :] = vec[:2]
                self.proj_wgt  [idx, i,  ] = 1
         
         
    def set_defocus(self,idx,def_file,skip_max_res=True):
        if( _is_ext(def_file,'defocus') ):
            line = _np.loadtxt(def_file,dtype=_np.float32,comments='#',max_rows=1)
            version = int(line[-1])
            if version == 2:
                self.def_U  [idx,0] = 10*line[4]
                self.def_V  [idx,0] = 10*line[4]
                self.def_ang[idx,0] = 0
                data = _np.loadtxt(def_file,dtype=_np.float32,comments='#',skiprows=1)
                n = data.shape[0]
                self.def_U  [idx,1:n+1] = 10*data[:,4]
                self.def_V  [idx,1:n+1] = 10*data[:,4]
                self.def_ang[idx,1:n+1] = 0
            elif version == 3:
                data = _np.loadtxt(def_file,dtype=_np.float32,comments='#',skiprows=1)
                n = data.shape[0]
                self.def_U  [idx,:n] = 10*data[:,4]
                self.def_V  [idx,:n] = 10*data[:,5]
                self.def_ang[idx,:n] = data[:,6]
            else:
                raise NameError('Invalid DEFOCUS format')
        elif _is_ext(def_file,'txt'):
            P = self.num_proj[idx]
            buffer = _np.loadtxt(def_file,dtype=_np.float32,comments='#',ndmin=2,max_rows=P)
            self.def_U     [idx,:P]   = buffer[:,0]
            self.def_V     [idx,:P]   = buffer[:,1]
            self.def_ang   [idx,:P]   = buffer[:,2]
            self.def_phas  [idx,:P]   = buffer[:,3]
            self.def_Bfct  [idx,:P]   = buffer[:,4]
            self.def_ExFl  [idx,:P]   = buffer[:,5]
            self.def_mres  [idx,:P]   = buffer[:,6]
            self.def_scor  [idx,:P]   = buffer[:,7]
            if skip_max_res:
                self.def_mres[idx,:P] = 0
        else:
            raise NameError('Invalid filename')
