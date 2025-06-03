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

__all__ = ['radial_average',
           'radial_expansion',
           'fsc_get',
           'fsc_analyse',
           'denoise_l0',
           'bandpass',
           'radial_blurring',
           'apply_FOM',
           'euDYN_rotm',
           'euZYZ_rotm',
           'rotm_euZYZ',
           'get_extension',
           'is_extension',
           'force_extension',
           'time_now',
           'create_sphere',
           'bin_vol',
          ]

import datetime
import susan.io.mrc as mrc
import numpy as np
from numba import jit
from scipy.ndimage import gaussian_filter
from os.path import splitext as split_ext
import susan.utils.datatypes as datatypes

###########################################

@jit(nopython=True,cache=True)
def radial_average(v):
    assert v.ndim == 3, "Volume must be three-dimensional"

    N = max( max(v.shape[0],v.shape[1]), v.shape[2] )+1
    val = np.zeros(N)
    wgt = np.zeros(N)

    cnt_z = v.shape[0]//2
    cnt_y = v.shape[1]//2
    cnt_x = v.shape[2]//2

    for k in range(v.shape[0]):
        z = k - cnt_z
        for j in range(v.shape[1]):
            y = j - cnt_y
            for i in range(v.shape[2]):
                x = i - cnt_x
                r = np.sqrt( x**2 + y**2 + z**2 )
                r = np.int32(np.round(r))
                if r < N:
                    val[r] += v[k,j,i]
                    wgt[r] += 1.0
    val = val/np.maximum(wgt,1)
    return val

@jit(nopython=True,cache=True)
def radial_expansion(arr):
    cnt = arr.shape[0]-1
    N = 2*cnt
    rslt = np.zeros((N,N,N),np.float32)

    for k in range(N):
        z = k - cnt
        for j in range(N):
            y = j - cnt
            for i in range(N):
                x = i - cnt
                r = np.sqrt( x**2 + y**2 + z**2 )
                r = np.int32(np.round(r))
                if r < cnt:
                    rslt[k,j,i] = arr[r]
    return rslt

###########################################

@jit(nopython=True)
def _get_weight(s_r,s_a):
    acc = 0
    for k in range(-2,3):
        for j in range(-2,3):
            for i in range(-2,3):
                rad = k
                arc = np.sqrt((i**2)+(j**2))
                blr = np.exp( -((arc/s_a)**2 + (rad/s_r)**2)/2 )
                acc += blr
    return acc

@jit(nopython=True)
def _apply_radial_blur(v_out,v_in,sigma,weight,min_rad,max_rad,sigma_shell):
    Nh = v_in.shape[0]//2
    s_r = sigma_shell
    for z in range(v_in.shape[0]):
        for y in range(v_in.shape[1]):
            for x in range(v_in.shape[2]):
                x_w = x-Nh
                y_w = y-Nh
                z_w = z-Nh
                n_w = np.sqrt((x_w**2)+(y_w**2)+(z_w**2))
                if (n_w >= min_rad) and (n_w <= max_rad):
                    s_index = int( np.round( 2*(n_w - min_rad) ) )
                    s_a = sigma[s_index]
                    s_w = weight[s_index]
                    acc = 0
                    for k in range(-2,3):
                        for j in range(-2,3):
                            for i in range(-2,3):
                                x_r = x_w + i
                                y_r = y_w + j
                                z_r = z_w + k
                                n_r = np.abs(np.sqrt((x_r**2)+(y_r**2)+(z_r**2)))
                                rad = n_r - n_w
                                if n_r < 1e-6: n_r = 1
                                if n_w < 1e-6: n_w = 1
                                ang = (x_w*x_r + y_w*y_r + z_w*z_r)/(n_r*n_w)
                                arc = n_r*np.abs(np.arccos( min(max(ang,-1),1) ))
                                blr = np.exp( -((arc/s_a)**2 + (rad/s_r)**2)/2 )
                                acc += blr * v_in[ z+k, y+j, x+i ]
                    v_out[z,y,x] = acc/s_w

def radial_blurring(v,max_sigma=1.0,min_rad=10,max_rad=None,min_sigma=0.4,sigma_shell=0.4):
    if max_rad is None:
        max_rad = v.shape[0]//4

    radii  = np.arange(min_rad,max_rad+0.5,0.5)
    sigma  = np.linspace(min_sigma,max_sigma,radii.size)
    weight = np.zeros_like(sigma)
    for i in range(sigma.shape[0]):
        weight[i] = _get_weight(sigma_shell,sigma[i])

    msk = create_sphere((max_rad+min_rad)/2,v.shape[0] )
    v_out = gaussian_filter(v,min_sigma)*msk + (1-msk)*gaussian_filter(v,max_sigma)
    _apply_radial_blur(v_out,v,sigma,weight,min_rad,max_rad,sigma_shell)
    return v_out

###########################################

class BayesianEstimator3D:
    def __init__(self, ini_vol: np.ndarray, ini_var: np.ndarray):
        self.post_avg = ini_vol
        self.post_var = ini_var

    def update(self, new_vol: np.ndarray, new_var: np.ndarray) -> np.ndarray:
        # Bayesian update (assuming Gaussian prior and likelihood):
        var = 1 / ( (1/self.post_var) + (1/new_var) )
        avg = var * ( (self.post_avg/self.post_var) + (new_vol/new_var) )

        # Update internal state
        self.post_avg = avg
        self.post_var = var

        return self.post_avg

###########################################

def denoise_l0(v,l0_lambda,rho=1,max_clip=-1):
    rho  = max(min(rho,1),0)
    th   = np.quantile(v,l0_lambda)
    rslt = np.copy(v)
    rslt[rslt>th] = 0
    if max_clip > 0:
        th   = np.quantile(rslt,max_clip)
        rslt = np.maximum(rslt,th)
    if rho < 1:
        rslt = rho*rslt + (1-rho)*v
    return rslt

###########################################

@jit(nopython=True,cache=True)
def _core_apply_fourier_rad_wgt(v_fou,wgt):
    c_z = v_fou.shape[0]/2
    c_y = v_fou.shape[1]/2
    for z in range(v_fou.shape[0]):
        Z = (z-c_z)**2
        for y in range(v_fou.shape[1]):
            Y = (y-c_y)**2
            for x in range(v_fou.shape[2]):
                X = x**2
                r = int(np.sqrt(X+Y+Z))
                r = min(r,wgt.shape[0]-1)
                v_fou[z,y,x] = wgt[r]*v_fou[z,y,x]

def _apply_fourier_rad_wgt(v,wgt):
    v_f = np.fft.fftshift(np.fft.rfftn(v),axes=(0,1))
    _core_apply_fourier_rad_wgt(v_f,wgt)
    rslt = np.fft.irfftn(np.fft.ifftshift(v_f,axes=(0,1)))
    rslt = np.float32(rslt)
    return rslt

def _gen_bandpass_wgt(box_size,lowpass,highpass=0,rolloff=1):
    t = np.arange(box_size//2+1)
    wgt = np.ones(t.shape,np.float32)
    
    rolloff = max(rolloff,1)
    if lowpass > 0:
        x = (t-lowpass)/rolloff
        x = np.pi*x.clip(0,1)
        m = 0.5*np.cos(x)+0.5
        wgt = wgt*m
    if highpass > 0:
        x = (highpass-t)/rolloff
        x = np.pi*x.clip(0,1)
        m = 0.5*np.cos(x)+0.5
        wgt = wgt*m
    return wgt

def bandpass(v,lowpass,highpass=0,rolloff=1):
    bp  = _gen_bandpass_wgt(v.shape[1],lowpass,highpass,rolloff)
    return _apply_fourier_rad_wgt(v,bp)

def apply_FOM(v,fsc_array):
    wgt = np.sqrt(fsc_array.clip(0,1))
    return _apply_fourier_rad_wgt(v,wgt)

###########################################

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
    if fpix == 0:
        res = 0
    else:
        res  = (2*(fsc.size-1)*apix)/fpix
    rslt = datatypes.fsc_info(fpix,res)
    return rslt

###########################################

@jit(nopython=True,cache=True)
def euDYN_rotm(R,eu):
    cos_theta = np.cos(eu[2])
    cos_phi   = np.cos(eu[1])
    cos_psi   = np.cos(eu[0])
    sin_theta = np.sin(eu[2])
    sin_phi   = np.sin(eu[1])
    sin_psi   = np.sin(eu[0])
    R[0,0] = cos_theta*cos_psi - cos_phi*sin_theta*sin_psi
    R[0,1] = -cos_theta*sin_psi - cos_phi*cos_psi*sin_theta
    R[0,2] = sin_theta*sin_phi
    R[1,0] = cos_psi*sin_theta + cos_theta*cos_phi*sin_psi
    R[1,1] = cos_theta*cos_phi*cos_psi - sin_theta*sin_psi
    R[1,2] = -cos_theta*sin_phi
    R[2,0] = sin_phi*sin_psi
    R[2,1] = cos_psi*sin_phi
    R[2,2] = cos_phi

@jit(nopython=True,cache=True)
def euZYZ_rotm(R,eu):
    #R = np.zeros((3,3))
    cos_theta = np.cos(eu[0])
    cos_phi = np.cos(eu[1])
    cos_psi = np.cos(eu[2])
    sin_theta = np.sin(eu[0])
    sin_phi = np.sin(eu[1])
    sin_psi = np.sin(eu[2])
    R[0,0] = cos_theta*cos_phi*cos_psi - sin_theta*sin_psi
    R[0,1] = -cos_psi*sin_theta - cos_theta*cos_phi*sin_psi
    R[0,2] = cos_theta*sin_phi
    R[1,0] = cos_theta*sin_psi + cos_phi*cos_psi*sin_theta
    R[1,1] = cos_theta*cos_psi - cos_phi*sin_theta*sin_psi
    R[1,2] = sin_theta*sin_phi
    R[2,0] = -cos_psi*sin_phi
    R[2,1] = sin_phi*sin_psi
    R[2,2] = cos_phi
    #return R

@jit(nopython=True,cache=True)
def rotm_euZYZ(euler, R):
    if np.abs(R[2,2]-1)<1e-6:
        euler[0] = 0
        euler[1] = 0
        euler[2] = np.arcsin(R[1,0])
    else:
        euler[0] = np.arctan2(R[1,2],R[0,2])
        euler[1] = np.arctan2(np.sqrt( 1-(R[2,2]*R[2,2]) ),R[2,2])
        euler[2] = np.arctan2(R[2,1],-R[2,0])

###########################################

def get_extension(filename):
    _,ext = split_ext(filename)
    return ext

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

###########################################

def time_now():
    return datetime.datetime.now()

###########################################

def create_sphere(r,N):
    M = N//2
    t = np.arange(-M,M)
    x,y,z = np.meshgrid(t,t,t)
    rad = np.sqrt( x**2 + y**2 + z**2 )
    return np.float32((r-rad).clip(0,1))

###########################################

def bin_vol(vol,bin_level):
    s = (2**bin_level)
    v = bandpass(vol,vol.shape[0]//s-1)
    v = v[::s,::s,::s]
    return np.float32(v)
