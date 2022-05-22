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

__all__ = ['read','write','get_info']

import numpy as _np

def read(filename):
    mrc_shape = _np.fromfile(filename,dtype=_np.uint32 ,count=3)
    mrc_mode  = _np.fromfile(filename,dtype=_np.uint32 ,count=1,offset=12)
    mrc_sampl = _np.fromfile(filename,dtype=_np.uint32 ,count=3,offset=28)
    mrc_cellA = _np.fromfile(filename,dtype=_np.float32,count=3,offset=40)
    mrc_offst = _np.fromfile(filename,dtype=_np.uint32 ,count=1,offset=92)

    pix_size  = (mrc_cellA/mrc_sampl).astype(_np.float32)

    if mrc_mode == 0:
        in_type = _np.int8
    elif mrc_mode == 1:
        in_type = _np.int16
    elif mrc_mode == 2:
        in_type = _np.float32
    elif mrc_mode == 6:
        in_type = _np.uint16
    elif mrc_mode == 12:
        in_type = np.float16
    else:
        raise ValueError

    data = _np.fromfile(filename,dtype=in_type,count=-1,offset=(1024+mrc_offst[0]))
    data = _np.reshape(data,(mrc_shape[2],mrc_shape[1],mrc_shape[0]))

    return data,pix_size

def get_info(filename):
    mrc_shape = _np.fromfile(filename,dtype=_np.uint32 ,count=3)
    mrc_mode  = _np.fromfile(filename,dtype=_np.uint32 ,count=1,offset=12)
    mrc_sampl = _np.fromfile(filename,dtype=_np.uint32 ,count=3,offset=28)
    mrc_cellA = _np.fromfile(filename,dtype=_np.float32,count=3,offset=40)
    
    pix_size  = (mrc_cellA/mrc_sampl).astype(_np.float32)

    if mrc_mode == 0:
        in_type = _np.int8
    elif mrc_mode == 1:
        in_type = _np.int16
    elif mrc_mode == 2:
        in_type = _np.float32
    elif mrc_mode == 6:
        in_type = _np.uint16
    elif mrc_mode == 12:
        in_type = np.float16
    else:
        raise ValueError

    return mrc_shape,pix_size,in_type

def write(data,filename,apix=1):
    apix = _np.array(apix,dtype=_np.float32)
    if apix.size == 1:
        apix = _np.array((apix,apix,apix))
    apix = apix*_np.array(data.shape,dtype=_np.float32)
    apix_uint32 = apix.view(_np.uint32)
    hdr  = _np.zeros(256,dtype=_np.uint32)
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

