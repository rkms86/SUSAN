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
from susan.utils import euZYZ_rotm as _euZYZ_rotm
from susan.utils import rotm_euZYZ as _rotm_euZYZ

class PtclsGeom:
    
###############################################################################
    @staticmethod
    def _validate_single_rotation(eZYZdeg,R):
        if eZYZdeg is not None and R is not None:
            raise ValueError('Set either eZYZdeg or R, not both.')
        
        if eZYZdeg is None and R is None:
            return None
        
        if eZYZdeg is not None:
        
            eu = _np.deg2rad(_np.array(eZYZdeg,dtype=_np.float32))
            if eu.size != 3:
                raise ValueError('eZYZdeg must be a 3-element array/vector.')
            R = _np.zeros((3,3),_np.float32)
            _euZYZ_rotm(R,eu)
        
        else:
            if R.ndim != 2 or R.shape[0] != 3 or R.shape[1] != 3:
                raise ValueError('R must be a 3x3 matrix.')
        
        return _np.float32(R)
    
    @staticmethod
    def _validate_single_translation(t):
        if t is None:
            t = _np.zeros(3,_np.float32)
        else:
            t = _np.array(t,_np.float32)
            if t.size != 3:
                raise ValueError('t must be a 3-element array/vector.')
        return t
    
    @staticmethod
    @_jit(nopython=True,cache=True)
    def _inplace_shift(ali_eZYZ,ali_t,t):
        R = _np.zeros((3,3),_np.float32)
        for i in range(ali_eZYZ.shape[0]):
            _euZYZ_rotm(R,ali_eZYZ[i])
            tout = R@t
            ali_t[i,:] = ali_t[i,:] + tout
            
    @staticmethod
    @_jit(nopython=True,cache=True)
    def _inplace_rot_shift(ali_eZYZ,ali_t,R,t):
        R_in = _np.zeros((3,3),_np.float32)
        Rout = _np.zeros((3,3),_np.float32)
        for i in range(ali_eZYZ.shape[0]):
            _euZYZ_rotm(R_in,ali_eZYZ[i])
            Rout = (R@R_in.transpose()).transpose()
            _rotm_euZYZ(ali_eZYZ[i],Rout)
            tout = Rout@t
            ali_t[i,:] = ali_t[i,:] + tout
    
    @staticmethod
    def rot_shift(ptcls,eZYZdeg=None,R=None,t=None,ref_idx=0):
        R = PtclsGeom._validate_single_rotation(eZYZdeg,R)
        t = PtclsGeom._validate_single_translation(t)
                
        if R is None:
            PtclsGeom._inplace_shift(ptcls.ali_eu[ref_idx],ptcls.ali_t[ref_idx],t)
        else:
            PtclsGeom._inplace_rot_shift(ptcls.ali_eu[ref_idx],ptcls.ali_t[ref_idx],R,t)
    
###############################################################################
    @staticmethod
    def _validate_multiple_rotations(eZYZdeg,R):
        if eZYZdeg is not None and R is not None:
            raise ValueError('Set either eZYZdeg or R, not both.')
        
        if eZYZdeg is None and R is None:
            return None
        
        if eZYZdeg is not None:
        
            eu = _np.deg2rad(_np.array(eZYZdeg,dtype=_np.float32))
            if eu.ndim == 1 and eu.size == 3:
                R = _np.zeros((1,3,3),_np.float32)
                _euZYZ_rotm(R[0],eu)
            elif eu.ndim == 2 and eu.shape[1] == 3:
                R = _np.zeros((eu.shape[0],3,3),_np.float32)
                for i in range(eu.shape[0]):
                    _euZYZ_rotm(R[i],eu[i])
            else:
                raise ValueError('eZYZdeg must be a 3-element array/vector or a stack of them.')
            
        
        else:
            if R.ndim < 2 or R.ndim > 3:
                raise ValueError('R must be a 3-by-3 matrix or a stack of them.')
            elif R.ndim == 2:
                R = R[_np.newaxis,:,:]
        
        return R

    @staticmethod
    def _validate_multiple_translations(t):
        if t is not None:
            t = _np.array(t,_np.float32)
            if t.ndim == 1:
                t = t[_np.newaxis,:]
            elif t.ndim != 2:
                raise ValueError('t must be a 1D or 2D matrix.')
        return t

    @staticmethod
    def _validate_multiple_inputs(R,t):
        if R is None and t is None:
            raise ValueError('Set the angles or the shifts...')
        
        if R is not None and t is not None:
            if R.shape[0] != t.shape[0]:
                raise ValueError('Number of angles do not match the number of shifts.')
        
        if R is None and t is not None:
            if t.shape[1] != 3:
                raise ValueError('t is not a N-by-3 matrix.')
        
        if R is not None and t is None:
            if R.shape[1] != 3 or R.shape[2] != 3:
                raise ValueError('R is not a N-by-3-by-3 matrix.')
            t = _np.zeros((R.shape[0],3),_np.float32)
        
        return R,t
    
    @staticmethod
    @_jit(nopython=True,cache=True)
    def _outplace_shift(out_ali_t,in_ali_eZYZ,in_ali_t,t):
        R = _np.zeros((3,3),_np.float32)
        out_ix = 0
        for i in range(in_ali_eZYZ.shape[0]):
            _euZYZ_rotm(R,in_ali_eZYZ[i])
            for j in range(t.shape[0]):
                tout = R@t[j]
                out_ali_t[out_ix,:] = in_ali_t[i,:] + tout
                out_ix = out_ix + 1

    @staticmethod
    @_jit(nopython=True,cache=True)
    def _outplace_rot_shift(out_ali_eZYZ,out_ali_t,in_ali_eZYZ,in_ali_t,R,t):
        R_in = _np.zeros((3,3),_np.float32)
        Rout = _np.zeros((3,3),_np.float32)
        out_ix = 0
        for i in range(in_ali_eZYZ.shape[0]):
            _euZYZ_rotm(R_in,in_ali_eZYZ[i])
            for j in range(t.shape[0]):
                Rout = (R[j]@R_in.transpose()).transpose()
                tout = Rout@t[j]
                out_ali_t[out_ix,:] = in_ali_t[i,:] + tout
                _rotm_euZYZ(out_ali_eZYZ[out_ix],Rout)
                out_ix = out_ix + 1

    @staticmethod
    def expand_by_rot_shift(ptcls,eZYZdeg=None,R=None,t=None,ref_idx=0):
        R   = PtclsGeom._validate_multiple_rotations(eZYZdeg,R)
        t   = PtclsGeom._validate_multiple_translations(t)
        R,t = PtclsGeom._validate_multiple_inputs(R,t)
        
        idx_expand = _np.tile(_np.arange(ptcls.n_ptcl),(t.shape[0],1)).transpose().flatten()
        ptcls_out = ptcls.select(idx_expand)
        
        if R is None:
            PtclsGeom._outplace_shift(ptcls_out.ali_t[ref_idx],ptcls.ali_eu[ref_idx],ptcls.ali_t[ref_idx],t)
        else:
            PtclsGeom._outplace_rot_shift(ptcls_out.ali_eu[ref_idx],ptcls_out.ali_t[ref_idx],ptcls.ali_eu[ref_idx],ptcls.ali_t[ref_idx],R,t)
        
        ptcls_out.update_position(ref_idx)
        return ptcls_out
        
###############################################################################
    @staticmethod
    @_jit(nopython=True,cache=True)
    def _enable_by_tilt(prj_w,tomo_cix,prj_eu,prj_wgt,tilt_min,tilt_max):
        for p in range(prj_w.shape[0]):
            t_id = tomo_cix[p]
            for k in range(prj_w.shape[1]):
                if prj_wgt[t_id,k] > 0:
                    tilt = _np.abs(prj_eu[t_id,k,1])
                    prj_w[p,k] = (tilt<tilt_max)&(tilt>=tilt_min)
                else:
                    prj_w[p,k] = 0
    
    @staticmethod
    def enable_by_tilt(ptcls,tomos,tilt_deg_max,tilt_deg_min=0):
        tilt_max = _np.abs(tilt_deg_max)
        tilt_min = _np.abs(tilt_deg_min)
        PtclsGeom._enable_by_tilt(ptcls.prj_w,ptcls.tomo_cix,tomos.proj_eZYZ,tomos.proj_wgt,tilt_min,tilt_max)

###############################################################################
    @staticmethod
    @_jit(nopython=True,cache=True)
    def _disable_closer(w_mask,pos,sort_ix,dist2):
        N = sort_ix.size
        for i_cur in range(N):
            idx_cur = sort_ix[i_cur]
            if w_mask[idx_cur]:
                for i_nxt in range(i_cur+1,N):
                    idx_nxt = sort_ix[i_nxt]
                    if w_mask[idx_nxt]:
                        vec = pos[idx_cur] - pos[idx_nxt]
                        d   = (vec*vec).sum()
                        if  d < dist2:
                            w_mask[idx_nxt] = False
        
    @staticmethod
    def discard_closer(ptcls,min_dist_angs,ref_idx=0,verbose=False):
        t_id = _np.unique( ptcls.tomo_cix )
        mask = _np.ones(ptcls.tomo_cix.shape,bool)
        dist = min_dist_angs*min_dist_angs
        
        if verbose:
            print('%d particles in %d tomograms. Processing:'%(ptcls.n_ptcl,t_id.size))
            
        for tid in t_id:
            t_mask  = ptcls.tomo_cix == tid
            cur_cc  = ptcls.ali_cc[ref_idx,t_mask]
            sort_ix = _np.argsort(cur_cc)[::-1]
            pos     = ptcls.position[t_mask] + ptcls.ali_t[ref_idx,t_mask]
            w_mask  = mask[t_mask]
            PtclsGeom._disable_closer(w_mask,pos,sort_ix,dist)
            mask[t_mask] = w_mask
            if verbose:
                print('\tTomogram index %3d: from %7d to %7d particles.'%(tid,sort_ix.size,w_mask.sum()))
        
        if verbose:
            print('Remaining particles: %d'%(mask.sum()))
        
        return ptcls.select( mask )
    
###############################################################################
    @staticmethod
    @_jit(nopython=True,cache=True)
    def _get_min_dist(dist,pos):
        N = pos.shape[0]
        for idx_cur in range(N):
            min_d   = -1
            for idx_wrk in range(N):
                if idx_cur != idx_wrk:
                    d = pos[idx_cur] - pos[idx_wrk]
                    d = _np.sqrt( (d*d).sum() )
                    if min_d < 0 or d < min_d:
                        min_d = d
            dist[idx_cur] = min_d
    
    @staticmethod
    def get_min_distance(ptcls,ref_idx=0):
        t_id = _np.unique( ptcls.tomo_cix )
        dist = _np.zeros(ptcls.tomo_cix.shape,_np.float32)
        
        for tid in t_id:
            t_mask  = ptcls.tomo_cix == tid
            pos     = ptcls.position[t_mask] + ptcls.ali_t[ref_idx,t_mask]
            d_mask  = dist[t_mask]
            PtclsGeom._get_min_dist(d_mask,pos)
            dist[t_mask] = d_mask
        
        return dist
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
