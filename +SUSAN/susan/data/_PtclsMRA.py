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

class PtclsMRA:
    
    @staticmethod
    def duplicate(ptcls,ref_idx=0):
        idx = _np.array(ref_idx)
        if idx.size == 1:
            ptcls.ali_eu = _np.concatenate((ptcls.ali_eu,ptcls.ali_eu[ref_idx][_np.newaxis,:,:]))
            ptcls.ali_t  = _np.concatenate((ptcls.ali_t ,ptcls.ali_t [ref_idx][_np.newaxis,:,:]))
            ptcls.ali_cc = _np.concatenate((ptcls.ali_cc,ptcls.ali_cc[ref_idx][_np.newaxis,:]  ))
            ptcls.ali_w  = _np.concatenate((ptcls.ali_w ,ptcls.ali_w [ref_idx][_np.newaxis,:]  ))
        else:
            ptcls.ali_eu = _np.concatenate((ptcls.ali_eu,ptcls.ali_eu[ref_idx,:,:]))
            ptcls.ali_t  = _np.concatenate((ptcls.ali_t ,ptcls.ali_t [ref_idx,:,:]))
            ptcls.ali_cc = _np.concatenate((ptcls.ali_cc,ptcls.ali_cc[ref_idx,:]))
            ptcls.ali_w  = _np.concatenate((ptcls.ali_w ,ptcls.ali_w [ref_idx,:]))
    
    @staticmethod
    def select_ref(ptcls,ref_idx):
        idx = _np.array(ref_idx)
        if idx.size == 1:
            rslt = ptcls.select( ptcls.ref_cix == idx )
            rslt.ali_eu = rslt.ali_eu[idx,:,:][_np.newaxis,:,:]
            rslt.ali_t  = rslt.ali_t [idx,:,:][_np.newaxis,:,:]
            rslt.ali_cc = rslt.ali_cc[idx,:]  [_np.newaxis,:]
            rslt.ali_w  = rslt.ali_w [idx,:]  [_np.newaxis,:]
            rslt.ref_cix[:] = 0
        else:
            mask = _np.zeros( ptcls.n_ptcl, bool )
            for i in range(idx.shape[0]):
                mask = mask | (ptcls.ref_cix == idx[i])
            rslt = ptcls.select( mask == True )
            rslt.ali_eu = rslt.ali_eu[idx,:,:]
            rslt.ali_t  = rslt.ali_t [idx,:,:]
            rslt.ali_cc = rslt.ali_cc[idx,:]
            rslt.ali_w  = rslt.ali_w [idx,:]
            for i in range(idx.shape[0]):
                rslt.ref_cix[ rslt.ref_cix==idx[i] ] = i
        return rslt

        
        
        
        
        
        
        
        
        
        
