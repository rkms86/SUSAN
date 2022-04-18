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

import numpy as np
from numba import jit

def read_mrc(filename):
    mrc_shape = np.fromfile(filename,dtype=np.uint32 ,count=3)
    mrc_mode  = np.fromfile(filename,dtype=np.uint32 ,count=1,offset=12)
    mrc_sampl = np.fromfile(filename,dtype=np.uint32 ,count=3,offset=28)
    mrc_cellA = np.fromfile(filename,dtype=np.float32,count=3,offset=40)
    mrc_offst = np.fromfile(filename,dtype=np.uint32 ,count=1,offset=92)

    pix_size  = mrc_cellA/mrc_sampl

    if mrc_mode == 0:
        in_type = np.int8
    elif mrc_mode == 1:
        in_type = np.int16
    elif mrc_mode == 2:
        in_type = np.float32
    elif mrc_mode == 6:
        in_type = np.uint16
    elif mrc_mode == 12:
        in_type = np.float16
    else:
        raise ValueError

    data = np.fromfile(filename,dtype=in_type,count=-1,offset=(1024+mrc_offst[0]))
    data = np.reshape(data,(mrc_shape[2],mrc_shape[1],mrc_shape[0]))

    return data,pix_size

def write_mrc(data,filename,apix=1):
    apix = np.array(apix,dtype=np.float32)
    if apix.size == 1:
        apix = np.array((apix,apix,apix))
    apix = apix*np.array(data.shape,dtype=np.float32)
    apix_uint32 = apix.view(np.uint32)
    hdr  = np.zeros(256,dtype=np.uint32)
    hdr[0]  = data.shape[2]
    hdr[1]  = data.shape[1]
    hdr[2]  = data.shape[0]
    hdr[3]  = 2
    hdr[7]  = data.shape[2]
    hdr[8]  = data.shape[1]
    hdr[9]  = data.shape[0]
    hdr[10]  = apix_uint32[2]
    hdr[11]  = apix_uint32[1]
    hdr[12]  = apix_uint32[0]
    hdr[13] = 1119092736 # 0x42b40000; // 90.0 in hexadecimal notation.
    hdr[14] = 1119092736 # 0x42b40000; // 90.0 in hexadecimal notation.
    hdr[15] = 1119092736 # 0x42b40000; // 90.0 in hexadecimal notation.
    hdr[16] = 1
    hdr[17] = 2
    hdr[18] = 3
    hdr[27] =      20140 # MRC2014 format
    hdr[52] =  542130509 # 0x2050414B ('MAP ')
    hdr[53] =      17476 # 0x00004444 little-endian
    f = open(filename,'w')
    hdr.tofile(f)
    data.tofile(f)
    f.close()

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
                r = np.int32(np.floor(r))
                if r < rslt.size:
                    rslt[r] = rslt[r] +  n[k,j,i]
                    tmp1[r] = tmp1[r] + d1[k,j,i]
                    tmp2[r] = tmp2[r] + d2[k,j,i]
    rslt = rslt/np.sqrt(tmp1*tmp2)
    return rslt


def fsc_get(v1,v2,msk=None):
    apix = 1
    if isinstance(v1,str):
        v1,apix = read_mrc(v1)
    
    if isinstance(v2,str):
        v2,_ = read_mrc(v2)
    
    if msk is not None:
        if isinstance(msk,str):
            msk,_ = read_mrc(msk)
        
        v1 = v1*msk
        v2 = v2*msk

    V1 = np.fft.fftshift( np.fft.rfftn(v1), axes=(0,1))
    V2 = np.fft.fftshift( np.fft.rfftn(v2), axes=(0,1))
    
    num = np.real(V1*np.conjugate(V2))
    d_1 = np.real(V1*np.conjugate(V1))
    d_2 = np.real(V2*np.conjugate(V2))
    
    fsc = _fsc_get_core(num,d_1,d_2)
    
    return fsc
    
    

