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
from numba import jit as _jit
from susan.data  import Tomograms       as _tomodef
from susan.utils import is_extension    as _is_ext
from susan.utils import force_extension as _force_ext
from susan.utils import euZYZ_rotm      as _euZYZ_rotm
from ._PtclsGeom import PtclsGeom       as _Geom
from ._PtclsMRA  import PtclsMRA        as _MRA

class Particles:
    
    Geom = _Geom
    MRA  = _MRA
    
    def __init__(self,filename=None,n_ptcl=0,n_proj=0,n_refs=0):
        if isinstance(filename, str):
            self.load(filename) 
        else:
            if n_ptcl > 0 and n_proj > 0 and n_refs>0:
                self.alloc(n_ptcl,n_proj,n_refs)
            else:
                raise NameError('Invalid input')

    def get_n_ptcl(self): return self.ptcl_id.shape[0]
    def get_n_refs(self): return self.ali_eu.shape[0]
    def get_n_proj(self): return self.prj_eu.shape[1]
    
    n_ptcl = property(get_n_ptcl)
    n_refs = property(get_n_refs)
    n_proj = property(get_n_proj)
    
    @staticmethod
    def _check_filename(filename):
        if not _is_ext(filename,'ptclsraw'):
            raise ValueError( 'Wrong file extension, do you mean ' + _force_ext(filename,'ptclsraw') + '?')
    
    def alloc(self,n_ptcl,n_proj,n_refs):
        self.ptcl_id  = _np.zeros( n_ptcl   ,dtype=_np.uint32 )
        self.tomo_id  = _np.zeros( n_ptcl   ,dtype=_np.uint32 )
        self.tomo_cix = _np.zeros( n_ptcl   ,dtype=_np.uint32 )
        self.position = _np.zeros((n_ptcl,3),dtype=_np.float32) # in Angstroms
        self.ref_cix  = _np.zeros( n_ptcl   ,dtype=_np.uint32 )
        self.half_id  = _np.zeros( n_ptcl   ,dtype=_np.uint32 )
        self.extra_1  = _np.zeros( n_ptcl   ,dtype=_np.float32)
        self.extra_2  = _np.zeros( n_ptcl   ,dtype=_np.float32)
        
        # 3D alignment
        self.ali_eu   = _np.zeros((n_refs,n_ptcl,3),dtype=_np.float32) # in Radians
        self.ali_t    = _np.zeros((n_refs,n_ptcl,3),dtype=_np.float32) # in Angstroms
        self.ali_cc   = _np.zeros((n_refs,n_ptcl  ),dtype=_np.float32)
        self.ali_w    = _np.zeros((n_refs,n_ptcl  ),dtype=_np.float32)
        
        # 2D alignment
        self.prj_eu   = _np.zeros((n_ptcl,n_proj,3),dtype=_np.float32) # in Radians
        self.prj_t    = _np.zeros((n_ptcl,n_proj,2),dtype=_np.float32) # in Angstroms
        self.prj_cc   = _np.zeros((n_ptcl,n_proj  ),dtype=_np.float32)
        self.prj_w    = _np.zeros((n_ptcl,n_proj  ),dtype=_np.float32)
        
        # Defocus
        self.def_U    = _np.zeros((n_ptcl,n_proj),dtype=_np.float32) # U (angstroms)
        self.def_V    = _np.zeros((n_ptcl,n_proj),dtype=_np.float32) # V (angstroms)
        self.def_ang  = _np.zeros((n_ptcl,n_proj),dtype=_np.float32) # angles (sexagesimal)
        self.def_phas = _np.zeros((n_ptcl,n_proj),dtype=_np.float32) # phase shift (sexagesimal?)
        self.def_Bfct = _np.zeros((n_ptcl,n_proj),dtype=_np.float32) # Bfactor
        self.def_ExFl = _np.zeros((n_ptcl,n_proj),dtype=_np.float32) # Exposure filter
        self.def_mres = _np.zeros((n_ptcl,n_proj),dtype=_np.float32) # Max. resolution (angstroms)
        self.def_scor = _np.zeros((n_ptcl,n_proj),dtype=_np.float32) # score

    def _load_header(self,fp):
        buffer = fp.read( 8 + 4*3 )
        if not _np.array_equal( buffer[:8], b'SsaPtcl1' ):
            raise NameError("Invalid File signature")
        return _np.frombuffer(buffer[8:],_np.uint32)
    
    def _parse_buffer(self,ix,buffer):
        self.ptcl_id[ix], self.tomo_id[ix], self.tomo_cix[ix] = buffer[:3].view(_np.uint32)
        self.position[ix] = buffer[3:6]
        self.ref_cix[ix], self.half_id[ix] = buffer[6:8].view(_np.uint32)
        self.extra_1[ix], self.extra_2[ix] = buffer[8:10]
        
        # 3D alignment
        R = self.ali_eu.shape[0]
        i=10;    self.ali_eu[:,ix,:] = buffer[i:i+3*R].reshape((R,3))
        i=i+3*R; self.ali_t [:,ix,:] = buffer[i:i+3*R].reshape((R,3))
        i=i+3*R; self.ali_cc[:,ix]   = buffer[i:i+  R].reshape((R))
        i=i+  R; self.ali_w [:,ix]   = buffer[i:i+  R].reshape((R))
        
        # 2D alignment
        P = self.prj_eu.shape[1]
        i=i+  R; self.prj_eu[ix,:,:] = buffer[i:i+3*P].reshape((P,3))
        i=i+3*P; self.prj_t [ix,:,:] = buffer[i:i+2*P].reshape((P,2))
        i=i+2*P; self.prj_cc[ix,:]   = buffer[i:i+  P]
        i=i+  P; self.prj_w [ix,:]   = buffer[i:i+  P]
        
        # Defocus
        i=i+  P; defocus = buffer[i:].reshape(P,8).transpose()
        self.def_U   [ix,:], self.def_V   [ix,:], self.def_ang[ix,:], \
        self.def_phas[ix,:], self.def_Bfct[ix,:], self.def_ExFl[ix,:], \
        self.def_mres[ix,:], self.def_scor[ix,:] = defocus
        
    def sort(self):
        idx = _np.lexsort((self.ptcl_id,self.tomo_id))
        self.ptcl_id  = self.ptcl_id [idx]
        self.tomo_id  = self.tomo_id [idx]
        self.tomo_cix = self.tomo_cix[idx]
        self.position = self.position[idx,:]
        self.ref_cix  = self.ref_cix [idx]
        self.half_id  = self.half_id [idx]
        self.extra_1  = self.extra_1 [idx]
        self.extra_2  = self.extra_2 [idx]
        # 3D alignment
        self.ali_eu   = self.ali_eu[:,idx,:]
        self.ali_t    = self.ali_t [:,idx,:]
        self.ali_cc   = self.ali_cc[:,idx]
        self.ali_w    = self.ali_w [:,idx]
        # 2D alignment
        self.prj_eu   = self.prj_eu[idx,:,:]
        self.prj_t    = self.prj_t [idx,:,:]
        self.prj_cc   = self.prj_cc[idx,:]
        self.prj_w    = self.prj_w [idx,:]
        # Defocus
        self.def_U    = self.def_U   [idx,:]
        self.def_V    = self.def_V   [idx,:]
        self.def_ang  = self.def_ang [idx,:]
        self.def_phas = self.def_phas[idx,:]
        self.def_Bfct = self.def_Bfct[idx,:]
        self.def_ExFl = self.def_ExFl[idx,:]
        self.def_mres = self.def_mres[idx,:]
        self.def_scor = self.def_scor[idx,:]

    def load(self,filename):
        Particles._check_filename(filename)
        
        fp = open(filename,"rb")
        n_ptcl, n_proj, n_refs = self._load_header(fp)
        self.alloc(n_ptcl,n_proj,n_refs)
        bytes_per_ptcl = 4*( 10 + 8*n_refs + 7*n_proj + 8*n_proj )
        for i in range(n_ptcl):
            buffer = _np.frombuffer(fp.read(bytes_per_ptcl),_np.float32)
            self._parse_buffer(i,buffer)
        fp.close()
    
    def save(self,filename):
        Particles._check_filename(filename)
        
        fp = open(filename,"wb")
        fp.write( b'SsaPtcl1' )
        _np.array( (self.n_ptcl,self.n_proj,self.n_refs), dtype=_np.uint32 ).tofile(fp)
        
        for ix in range(self.n_ptcl):
            _np.array( (self.ptcl_id[ix],self.tomo_id[ix],self.tomo_cix[ix]), dtype=_np.uint32 ).tofile(fp)
            self.position[ix,:].tofile(fp)
            _np.array( (self.ref_cix[ix],self.half_id[ix]), dtype=_np.uint32  ).tofile(fp)
            _np.array( (self.extra_1[ix],self.extra_2[ix]), dtype=_np.float32 ).tofile(fp)
            
            # 3D Alignment
            self.ali_eu[:,ix,:].flatten().tofile(fp)
            self.ali_t [:,ix,:].flatten().tofile(fp)
            self.ali_cc[:,ix]  .flatten().tofile(fp)
            self.ali_w [:,ix]  .flatten().tofile(fp)
            
            # 2D Alignment
            self.prj_eu[ix,:,:].flatten().tofile(fp)
            self.prj_t [ix,:,:].flatten().tofile(fp)
            self.prj_cc[ix,:]  .flatten().tofile(fp)
            self.prj_w [ix,:]  .flatten().tofile(fp)
            
            # Defocus
            defocus = _np.stack((self.def_U   [ix,:], self.def_V   [ix,:], self.def_ang[ix,:],
                                 self.def_phas[ix,:], self.def_Bfct[ix,:], self.def_ExFl[ix,:],
                                 self.def_mres[ix,:], self.def_scor[ix,:])).transpose()
            defocus.flatten().tofile(fp)
        fp.close()
    
    def __getitem__(self,idx):
        if isinstance(idx,slice):
            idx = _np.arange(idx.start,idx.stop,idx.step)
        return self.select(idx)
    
    def select(self,idx):
        idx = _np.array(idx)
        if( idx.ndim == 2 ):
            idx = idx[:,0]
        tmp = self.ptcl_id[idx]
        ptcls_out = Particles(n_ptcl=tmp.shape[0],n_proj=self.n_proj,n_refs=self.n_refs)
        ptcls_out.ptcl_id  = self.ptcl_id [idx]
        ptcls_out.tomo_id  = self.tomo_id [idx]
        ptcls_out.tomo_cix = self.tomo_cix[idx]
        ptcls_out.position = self.position[idx,:]
        ptcls_out.ref_cix  = self.ref_cix [idx]
        ptcls_out.half_id  = self.half_id [idx]
        ptcls_out.extra_1  = self.extra_1 [idx]
        ptcls_out.extra_2  = self.extra_2 [idx]
        # 3D alignment
        ptcls_out.ali_eu   = self.ali_eu[:,idx,:]
        ptcls_out.ali_t    = self.ali_t [:,idx,:]
        ptcls_out.ali_cc   = self.ali_cc[:,idx]
        ptcls_out.ali_w    = self.ali_w [:,idx]
        # 2D alignment
        ptcls_out.prj_eu   = self.prj_eu[idx,:,:]
        ptcls_out.prj_t    = self.prj_t [idx,:,:]
        ptcls_out.prj_cc   = self.prj_cc[idx,:]
        ptcls_out.prj_w    = self.prj_w [idx,:]
        # Defocus
        ptcls_out.def_U    = self.def_U   [idx,:]
        ptcls_out.def_V    = self.def_V   [idx,:]
        ptcls_out.def_ang  = self.def_ang [idx,:]
        ptcls_out.def_phas = self.def_phas[idx,:]
        ptcls_out.def_Bfct = self.def_Bfct[idx,:]
        ptcls_out.def_ExFl = self.def_ExFl[idx,:]
        ptcls_out.def_mres = self.def_mres[idx,:]
        ptcls_out.def_scor = self.def_scor[idx,:]
        # Sort
        ptcls_out.sort()
        return ptcls_out
    
    def append_ptcls(self,ptcls):
        self.ptcl_id  = _np.concatenate( (self.ptcl_id ,ptcls.ptcl_id ),axis=0 )
        self.tomo_id  = _np.concatenate( (self.tomo_id ,ptcls.tomo_id ),axis=0 )
        self.tomo_cix = _np.concatenate( (self.tomo_cix,ptcls.tomo_cix),axis=0 )
        self.position = _np.concatenate( (self.position,ptcls.position),axis=0 )
        self.ref_cix  = _np.concatenate( (self.ref_cix ,ptcls.ref_cix ),axis=0 )
        self.half_id  = _np.concatenate( (self.half_id ,ptcls.half_id ),axis=0 )
        self.extra_1  = _np.concatenate( (self.extra_1 ,ptcls.extra_1 ),axis=0 )
        self.extra_2  = _np.concatenate( (self.extra_2 ,ptcls.extra_2 ),axis=0 )
        # 3D alignment
        self.ali_eu   = _np.concatenate( (self.ali_eu,ptcls.ali_eu),axis=1 )
        self.ali_t    = _np.concatenate( (self.ali_t ,ptcls.ali_t ),axis=1 )
        self.ali_cc   = _np.concatenate( (self.ali_cc,ptcls.ali_cc),axis=1 )
        self.ali_w    = _np.concatenate( (self.ali_w ,ptcls.ali_w ),axis=1 )
        # 2D alignment
        self.prj_eu   = _np.concatenate( (self.prj_eu,ptcls.prj_eu),axis=0 )
        self.prj_t    = _np.concatenate( (self.prj_t ,ptcls.prj_t ),axis=0 )
        self.prj_cc   = _np.concatenate( (self.prj_cc,ptcls.prj_cc),axis=0 )
        self.prj_w    = _np.concatenate( (self.prj_w ,ptcls.prj_w ),axis=0 )
        # Defocus
        self.def_U    = _np.concatenate( (self.def_U   ,ptcls.def_U   ),axis=0 )
        self.def_V    = _np.concatenate( (self.def_V   ,ptcls.def_V   ),axis=0 )
        self.def_ang  = _np.concatenate( (self.def_ang ,ptcls.def_ang ),axis=0 )
        self.def_phas = _np.concatenate( (self.def_phas,ptcls.def_phas),axis=0 )
        self.def_Bfct = _np.concatenate( (self.def_Bfct,ptcls.def_Bfct),axis=0 )
        self.def_ExFl = _np.concatenate( (self.def_ExFl,ptcls.def_ExFl),axis=0 )
        self.def_mres = _np.concatenate( (self.def_mres,ptcls.def_mres),axis=0 )
        self.def_scor = _np.concatenate( (self.def_scor,ptcls.def_scor),axis=0 )
        # Sort
        self.sort()

    def set_weights(self,in_wgt):
        mask = (self.prj_w > 0).transpose()
        mask = mask * in_wgt
        self.prj_w[:,:] = mask.transpose()
        
    def halfsets_by_Y(self):
        tomo_ids = _np.unique( self.tomo_id )
        for tid in tomo_ids:
            idx = self.tomo_id == tid
            self.half_id[idx] = 1
            th = _np.quantile( self.position[idx,1].flatten() ,0.5)
            self.half_id[idx] = self.half_id[idx] + (self.position[idx,1]>th)

    def halfsets_even_odd(self):
        self.half_id[0::2] = 1
        self.half_id[1::2] = 2
        
    def update_position(self,ref_id=0):
        self.position = self.position + self.ali_t[ref_id]
        self.ali_t[ref_id,:,:] = 0

    @staticmethod
    @_jit(nopython=True,cache=True)
    def _update_new_defocus(dU,dV,R,p,num_proj,z_coef,dU_in,dV_in):
        for i in range(num_proj):
            dZ = R[i,2,0]*p[0] + R[i,2,1]*p[1] + R[i,2,2]*p[2]
            dU[i] = dU_in[i] + z_coef*dZ
            dV[i] = dV_in[i] + z_coef*dZ

    def update_defocus(self,tomos_info,ref_id=0,z_sign=-1):
        flag = True
        
        # Calculate tilt rotation matrix
        R_arr = _np.zeros((tomos_info.n_tomos,tomos_info.n_projs,3,3),dtype=_np.float32)
        for t in range(tomos_info.n_tomos):
            for p in range(tomos_info.num_proj[t]):
                _euZYZ_rotm(R_arr[t,p],_np.deg2rad(tomos_info.proj_eZYZ[t,p]))
        
        # Update defocus
        for k in range(self.n_ptcl):
            tid = self.tomo_cix[k]
            
            self.prj_w   [k,:] = tomos_info.proj_wgt[tid,:]
            self.def_ang [k,:] = tomos_info.def_ang [tid,:]
            self.def_phas[k,:] = tomos_info.def_phas[tid,:]
            self.def_Bfct[k,:] = tomos_info.def_Bfct[tid,:]
            self.def_ExFl[k,:] = tomos_info.def_ExFl[tid,:]
            self.def_mres[k,:] = tomos_info.def_mres[tid,:]
            self.def_scor[k,:] = tomos_info.def_scor[tid,:]
            
            # Note: Numba makes it ~23.3 times faster
            pos = self.position[k] + self.ali_t[ref_id,k]
            Particles._update_new_defocus(self.def_U[k],self.def_V[k],R_arr[tid],pos,tomos_info.num_proj[tid],z_sign,tomos_info.def_U[tid],tomos_info.def_V[tid])

    def x(self,ref_idx=0):
        return self.position[:,0] + self.ali_t[ref_idx,:,0]

    def y(self,ref_idx=0):
        return self.position[:,1] + self.ali_t[ref_idx,:,1]

    def z(self,ref_idx=0):
        return self.position[:,2] + self.ali_t[ref_idx,:,2]

    def pos(self,ref_idx=0):
        return self.position + self.ali_t[ref_idx]

    @staticmethod
    def _get_tomo_limit_angstroms(tomo_size,tomo_apix,border):
        return tomo_apix*( tomo_size-border )/2

    @staticmethod
    def _validate_tomogram(tomograms):
        if not isinstance(tomograms,_tomodef.Tomograms):
            raise ValueError('Tomograms must be a Tomograms object.')
        apix = _np.unique( tomograms.pix_size )
        if apix.size != 1:
            raise ValueError('Tomograms must have the same pixel size.')
        return apix[0]

    @staticmethod
    def _get_grid_step(s_ang,s_pix,apix):
        if   s_ang is     None and s_pix is     None:
            raise ValueError('Set the steps either in angstroms or pixels')
        elif s_ang is not None and s_pix is not None:
            raise ValueError('Set step_angstroms or step_pixels, not both.')
        elif s_ang is not None and s_pix is     None:
            return s_ang
        else:
            return s_pix*apix

    @staticmethod
    def _get_border_pixels(pix):
        p = _np.array(pix,_np.int32)
        if p.size == 1:
            return _np.array((pix,pix,pix),_np.int32)
        elif p.size == 3:
            return p
        else:
            raise ValueError('skip_border_pixels must be either a scalar or a 3-element vector')

    @staticmethod
    def grid_2d(tomograms,step_angstroms=None,step_pixels=None,skip_border_pixels=0):
        apix = Particles._validate_tomogram(tomograms)
        step = Particles._get_grid_step(step_angstroms,step_pixels,apix)
        brdr = Particles._get_border_pixels(skip_border_pixels)
        
        pts = _np.zeros((0,3),dtype=_np.float32)
        tcx = _np.zeros((0),dtype=_np.uint32)
        tid = _np.zeros((0),dtype=_np.uint32)
        for i in range( tomograms.n_tomos ):
            tomo_range = Particles._get_tomo_limit_angstroms(tomograms.tomo_size[i],tomograms.pix_size[i],brdr)
            t_x = _np.arange(0,tomo_range[0],step,dtype=_np.float32)
            t_x = _np.concatenate( (-t_x[::-1],t_x[1:]),dtype=_np.float32 )
            t_y = _np.arange(0,tomo_range[1],step,dtype=_np.float32)
            t_y = _np.concatenate( (-t_y[::-1],t_y[1:]),dtype=_np.float32 )
            x,y,z = _np.meshgrid(t_x,t_y,(0))
            pos = _np.stack( (x.flatten(),y.flatten(),z.flatten()), ).transpose()
            pts = _np.concatenate( (pts,pos),dtype=_np.float32 )
            tcx = _np.concatenate( (tcx,_np.repeat(_np.uint32(i),pos.shape[0])),dtype=_np.uint32 )
            tid = _np.concatenate( (tid,_np.repeat(tomograms.tomo_id[i],pos.shape[0])),dtype=_np.uint32 )
        
        ptcls = Particles(n_ptcl=pts.shape[0],n_proj=tomograms.n_projs,n_refs=1)
        ptcls.ptcl_id[:]    = _np.arange(1,pts.shape[0]+1,dtype=_np.uint32)
        ptcls.position[:,:] = pts
        ptcls.tomo_cix[:]   = tcx
        ptcls.tomo_id [:]   = tid
        ptcls.update_defocus(tomograms)
        return ptcls
    
    @staticmethod
    def grid_3d(tomograms,step_angstroms=None,step_pixels=None,skip_border_pixels=0):
        apix = Particles._validate_tomogram(tomograms)
        step = Particles._get_grid_step(step_angstroms,step_pixels,apix)
        brdr = Particles._get_border_pixels(skip_border_pixels)
        
        pts = _np.zeros((0,3),dtype=_np.float32)
        tcx = _np.zeros((0),dtype=_np.uint32)
        tid = _np.zeros((0),dtype=_np.uint32)
        for i in range( tomograms.n_tomos ):
            tomo_range = Particles._get_tomo_limit_angstroms(tomograms.tomo_size[i],tomograms.pix_size[i],brdr)
            t_x = _np.arange(0,tomo_range[0],step,dtype=_np.float32)
            t_x = _np.concatenate( (-t_x[::-1],t_x[1:]),dtype=_np.float32 )
            t_y = _np.arange(0,tomo_range[1],step,dtype=_np.float32)
            t_y = _np.concatenate( (-t_y[::-1],t_y[1:]),dtype=_np.float32 )
            t_z = _np.arange(0,tomo_range[2],step,dtype=_np.float32)
            t_z = _np.concatenate( (-t_z[::-1],t_z[1:]),dtype=_np.float32 )
            x,y,z = _np.meshgrid(t_x,t_y,t_z)
            pos = _np.stack( (x.flatten(),y.flatten(),z.flatten()), ).transpose()
            pts = _np.concatenate( (pts,pos),dtype=_np.float32 )
            tcx = _np.concatenate( (tcx,_np.repeat(_np.uint32(i),pos.shape[0])),dtype=_np.uint32 )
            tid = _np.concatenate( (tid,_np.repeat(tomograms.tomo_id[i],pos.shape[0])),dtype=_np.uint32 )
        
        ptcls = Particles(n_ptcl=pts.shape[0],n_proj=tomograms.n_projs,n_refs=1)
        ptcls.ptcl_id[:]    = _np.arange(1,pts.shape[0]+1,dtype=_np.uint32)
        ptcls.position[:,:] = pts
        ptcls.tomo_cix[:]   = tcx
        ptcls.tomo_id [:]   = tid
        ptcls.halfsets_even_odd()
        ptcls.update_defocus(tomograms)
        return ptcls



