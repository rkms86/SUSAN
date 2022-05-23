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

__all__ = ['fsc_get',
           'fsc_analyse',
           'denoise_l0',
           'bandpass',
           'euZYZ_rotm',
           'rotm_euZYZ',
           'is_extension',
           'force_extension',
           'time_now',
           'create_sphere',
           'bin_vol']

import datetime
import susan.io.mrc as mrc
import numpy as np
from numba import jit
from os.path import splitext as split_ext
import susan.utils.datatypes as datatypes

def denoise_l0(data,l0_lambda,rho=1):
    rho  = max(min(rho,1),0)
    th   = np.quantile(data,l0_lambda)
    rslt = np.minimum(data-th,0)
    if rho < 1:
        rslt = rho*rslt + (1-rho)*data
    return rslt
    
def bandpass(v,bp):
    bp = np.array(bp,dtype=np.float32)
    if bp.size == 1:
        bp = np.array((0,bp,1),dtype=np.float32)
    elif bp.size == 2:
        bp = np.array((bp[0],bp[1],1),dtype=np.float32)
    if bp[2] < 1:
        bp[2] = 1
    v_f = np.fft.rfftn(v)
    dim = np.array(v.shape,dtype=np.int32)//2
    x,y,z = np.meshgrid(np.arange(-dim[0],dim[0]),np.arange(-dim[1],dim[1]),np.arange(0,dim[2]+1))
    rad = np.sqrt( x**2 + y**2 + z**2 )
    l_p = 1-(rad-bp[1])/bp[2]
    np.clip(l_p,0,1,out=l_p)
    if bp[0] > 0:
        h_p = (rad-bp[1]-bp[2])/bp[2]
        np.clip(h_p,0,1,out=h_p)
        l_p = l_p*h_p
    l_p = np.fft.fftshift(l_p,axes=(0,1))
    v_f = v_f*l_p
    rslt = np.fft.irfftn(v_f)
    return rslt

@jit(nopython=True,cache=True)
def _fsc_get_core(n,d1,d2):
    rslt = np.zeros(n.shape[2],dtype=n.dtype)
    tmp1 = np.zeros(n.shape[2],dtype=n.dtype)
    tmp2 = np.zeros(n.shape[2],dtype=n.dtype)

    for k in range(n.shape[0]):
        z = k - n.shape[0]//2
        for j in range(n.shape[1]):
            y = j - n.shape[1]//2
            for i in range(n.shape[2]):
                r = np.sqrt( i**2 + y**2 + z**2 )
                r = np.int32(np.round(r))
                if r < rslt.size:
                    rslt[r] = rslt[r] +  n[k,j,i]
                    tmp1[r] = tmp1[r] + d1[k,j,i]
                    tmp2[r] = tmp2[r] + d2[k,j,i]
    rslt = rslt/np.sqrt(tmp1*tmp2)
    return rslt

def fsc_get(v1,v2,msk=None):
    apix = 1
    if isinstance(v1,str):
        v1,apix = mrc.read(v1)
    
    if isinstance(v2,str):
        v2,_ = mrc.read(v2)
    
    if msk is not None:
        if isinstance(msk,str):
            msk,_ = mrc.read(msk)
        
        v1 = v1*msk
        v2 = v2*msk

    V1 = np.fft.fftshift( np.fft.rfftn(v1), axes=(0,1))
    V2 = np.fft.fftshift( np.fft.rfftn(v2), axes=(0,1))
    
    num = np.real(V1*np.conjugate(V2)).astype(np.float32)
    d_1 = np.real(V1*np.conjugate(V1)).astype(np.float32)
    d_2 = np.real(V2*np.conjugate(V2)).astype(np.float32)
    
    fsc = _fsc_get_core(num,d_1,d_2)
    
    return fsc

def fsc_analyse(fsc,apix=1.0,thres=0.143):
    apix = np.array(apix)
    if( apix.size > 1 ):
        apix = apix[0]
    fpix = np.argwhere(fsc<thres)
    if fpix.size > 0:
        fpix = fpix[0,0]
    else:
        fpix = fsc.size-1
    res  = (2*(fsc.size-1)*apix)/fpix
    rslt = datatypes.fsc_info(fpix,res)
    return rslt

@jit(nopython=True,cache=True)
def euZYZ_rotm(R,eu):
    c1 = np.cos(eu[0])
    c2 = np.cos(eu[1])
    c3 = np.cos(eu[2])
    s1 = np.sin(eu[0])
    s2 = np.sin(eu[1])
    s3 = np.sin(eu[2])
    R[0,0] = c1*c2*c3 - s1*s3
    R[0,1] = -c3*s1 - c1*c2*s3
    R[0,2] = c1*s2
    R[1,0] = c1*s3 + c2*c3*s1
    R[1,1] = c1*c3 - c2*s1*s3
    R[1,2] = s1*s2
    R[2,0] = -c3*s2
    R[2,1] = s2*s3
    R[2,2] = c2

@jit(nopython=True,cache=True)
def rotm_euZYZ(eu,R):
    eu[1] = np.arccos( R[2,2] )
    if np.abs(R[2,0]) < 1e-5:
        # gimbal lock
        eu[0] = 0
        eu[2] = np.arctan2(R[0,1],R[1,1])
    else:
        eu[0] = np.arctan2(R[1,2], R[0,2])
        eu[2] = np.arctan2(R[2,1],-R[2,0])

def is_extension(filename,extension):
    _,ext = split_ext(filename)
    if( extension[0] == '.' ):
        return ext == extension
    else:
        return ext == '.'+extension

def force_extension(filename,extension):
    base,ext = split_ext(filename)
    new_ext = extension
    if new_ext[0] != '.':
        new_ext = '.' + extension
    return base + new_ext

def time_now():
    return datetime.datetime.now()

def create_sphere(radius,box_size):
    N = box_size//2
    t = np.arange(-N,N)
    x,y,z = np.meshgrid(t,t,t)
    rad = np.sqrt( x**2 + y**2 + z**2 )
    return np.float32((radius-rad).clip(0,1))

def bin_vol(data,bin_level):
    s = (2**bin_level)
    v = bandpass(data,data.shape[0]//s-1)
    v = v[::s,::s,::s]
    return np.float32(v)

    
    
    
    
    
    
    
    
