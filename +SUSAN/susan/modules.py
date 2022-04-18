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

import os,susan,math

def get_gpu_str(list_gpus_ids):
    gpu_str = ''
    for i in range(len(list_gpus_ids)):
        gpu_str = gpu_str + '%d,' % list_gpus_ids[i]
    gpu_str = gpu_str[0:-1]
    return gpu_str

def denoise_l0(map_name,lambda_val,rho_val):
    args =        ' -map_file '   + map_name
    args = args + ' -lambda %f' % lambda_val
    args = args + ' -rho %f' % rho_val
    cmd = os.path.dirname(susan.__file__)+'/../bin/susan_denoise_l0'
    cmd = cmd + args
    rslt = os.system(cmd)
    if not rslt == 0:
        raise NameError('Error executing the denoiser: ' + cmd)

class aligner:
    def __init__(self):
        self.list_gpus_ids     = [0]
        self.threads_per_gpu   = 1
        self.bandpass_highpass = 0
        self.bandpass_lowpass  = 20
        self.bandpass_rolloff  = 3
        self.dimensionality    = 3
        self.extra_padding     = 0
        self.allow_drift       = True
        self.halfsets_independ = False
        self.cone_range        = 0
        self.cone_step         = 1
        self.inplane_range     = 0
        self.inplane_step      = 1
        self.refine_level      = 0
        self.refine_factor     = 1
        self.offset_range      = [4,4,4]
        self.offset_step       = 1
        self.offset_type       = 'ellipsoid'
        self.padding_type      = 'noise'
        self.normalize_type    = 'zero_mean'
        self.ctf_correction    = 'on_reference'
        self.pseudo_symmetry   = 'c1'
        self.ssnr_S            = 0
        self.ssnr_F            = 0
        
    def set_angular_search(self,c_r=0,c_s=1,i_r=0,i_s=1):
        self.cone_range    = c_r
        self.cone_step     = c_s
        self.inplane_range = i_r
        self.inplane_step  = i_s
        
    def set_offset_search(self,off_range,off_step,off_type='ellipsoid'):
        if not off_type in ['ellipsoid','cylinider']:
            raise NameError('Invalid offset type. Only "ellipsoid" or "cylinder" are valid')
        self.offset_range = off_range
        self.offset_step  = off_step
        self.offset_type  = off_type
        
    def validate(self):
        if not self.dimensionality in [2,3]:
            raise NameError('Invalid dimensionality type. Only 3 or 3 are valid')
        
        if not self.offset_type in ['ellipsoid','cylinider']:
            raise NameError('Invalid offset type. Only "ellipsoid" or "cylinder" are valid')
        
        if not self.padding_type in ['zero','noise']:
            raise NameError('Invalid padding type. Only "zero" or "noise" are valid')
        
        if not self.normalize_type in ['none','zero_mean','zero_mean_one_std','zero_mean_proj_weight']:
            raise NameError('Invalid normalization type. Only "none", "zero_mean", "zero_mean_one_std" or "zero_mean_proj_weight" are valid')
        
        if not self.ctf_correction in ['none','on_reference','on_substack','wiener_ssnr','wiener_white','cfsc']:
            raise NameError('Invalid ctf correction type. Only "none", "on_reference", "on_substack", "wiener_ssnr", "wiener_white" or "cfsc" are valid')
            
    def get_args(self,ptcls_out,refs_file,tomos_file,ptcls_in,box_size):
        self.validate()
        n_threads = '%d' % (len(self.list_gpus_ids)*self.threads_per_gpu)
        gpu_str = get_gpu_str(self.list_gpus_ids)
        args =        ' -tomos_file ' + tomos_file
        args = args + ' -ptcls_in '   + ptcls_in
        args = args + ' -ptcls_out '  + ptcls_out
        args = args + ' -refs_file '  + refs_file
        args = args + ' -n_threads '  + n_threads
        args = args + ' -gpu_list '   + gpu_str
        args = args + ' -box_size %d' % box_size
        args = args + ' -pad_size %d' % self.extra_padding
        args = args + ' -pad_type '   + self.padding_type
        args = args + ' -norm_type '  + self.normalize_type
        args = args + ' -ctf_type '   + self.ctf_correction
        args = args + ' -ssnr_param ' + ('%f' % self.ssnr_F) + (',%f' % self.ssnr_S)
        args = args + ' -bandpass '   + ('%f' % self.bandpass_highpass) + (',%f' % self.bandpass_lowpass)
        args = args + ' -rolloff_f '  + ('%f' % self.bandpass_rolloff)
        args = args + ' -p_symmetry ' + self.pseudo_symmetry
        if self.halfsets_independ:
            args = args + ' -ali_halves 1'
        else:
            args = args + ' -ali_halves 0'
        if self.allow_drift:
            args = args + ' -allow_drift 1'
        else:
            args = args + ' -allow_drift 0'
        args = args + ' -cone '       + ('%f' % self.cone_range) + (',%f' % self.cone_step)
        args = args + ' -inplane '    + ('%f' % self.inplane_range) + (',%f' % self.inplane_step)
        args = args + ' -refine '     + ('%f' % self.refine_factor) + (',%f' % self.refine_level)
        args = args + ' -off_type '   + self.offset_type
        args = args + ' -off_params ' + ('%f' % self.offset_range[0]) + (',%f' % self.offset_range[1]) + (',%f' % self.offset_range[2]) + (',%f' % self.offset_step)
        args = args + ' -type %d'     % self.dimensionality
        return args
    
    def align(self,ptcls_out,refs_file,tomos_file,ptcls_in,box_size):
        cmd = os.path.dirname(susan.__file__)+'/../bin/susan_aligner'
        cmd = cmd + self.get_args(ptcls_out, refs_file, tomos_file, ptcls_in, box_size)
        rslt = os.system(cmd)
        if not rslt == 0:
            raise NameError('Error executing the alignment: ' + cmd)
    
    def align_mpi(self,ptcls_out,refs_file,tomos_file,ptcls_in,box_size,mpi_nodes):
        cmd = 'srun --mpi=pmi2 -n ' + ('%d ' % mpi_nodes)
        cmd = cmd + os.path.dirname(susan.__file__)+'/../bin/susan_aligner_mpi'
        cmd = cmd + self.get_args(ptcls_out, refs_file, tomos_file, ptcls_in, box_size)
        rslt = os.system(cmd)
        if not rslt == 0:
            raise NameError('Error executing the alignment: ' + cmd)
        
class averager:
    def __init__(self):
        self.list_gpus_ids     = [0]
        self.threads_per_gpu   = 1
        self.bandpass_highpass = 0
        self.bandpass_lowpass  = 20
        self.bandpass_rolloff  = 3
        self.extra_padding     = 0
        self.rec_halfsets      = False
        self.padding_type      = 'noise'
        self.normalize_type    = 'zero_mean'
        self.ctf_correction    = 'wiener'
        self.symmetry          = 'c1'
        self.ssnr_S            = 0
        self.ssnr_F            = 0
        self.inversion_iter    = 10
        self.inversion_gstd    = 0.75
        
    def validate(self):
        if not self.padding_type in ['zero','noise']:
            raise NameError('Invalid padding type. Only "zero" or "noise" are valid')
        
        if not self.normalize_type in ['none','zero_mean','zero_mean_one_std','zero_mean_proj_weight']:
            raise NameError('Invalid normalization type. Only "none", "zero_mean", "zero_mean_one_std" or "zero_mean_proj_weight" are valid')
        
        if not self.ctf_correction in ['none','phase_flip','wiener','wiener_ssnr']:
            raise NameError('Invalid ctf correction type. Only "none", "phase_flip", "wiener" ot "wiener_ssnr" are valid')
            
    def get_args(self,out_pfx,tomos_file,ptcls_in,box_size):
        self.validate()
        n_threads = '%d' % (len(self.list_gpus_ids)*self.threads_per_gpu)
        gpu_str = get_gpu_str(self.list_gpus_ids)
        args =        ' -tomos_file ' + tomos_file
        args = args + ' -out_prefix ' + out_pfx
        args = args + ' -ptcls_file ' + ptcls_in
        args = args + ' -n_threads '  + n_threads
        args = args + ' -gpu_list '   + gpu_str
        args = args + ' -box_size %d' % box_size
        args = args + ' -pad_size %d' % self.extra_padding
        args = args + ' -pad_type '   + self.padding_type
        args = args + ' -norm_type '  + self.normalize_type
        args = args + ' -ctf_type '   + self.ctf_correction
        args = args + ' -ssnr_param ' + ('%f' % self.ssnr_F) + (',%f' % self.ssnr_S)
        args = args + ' -w_inv_iter ' + ('%d' % self.inversion_iter)
        args = args + ' -w_inv_gstd ' + ('%f' % self.inversion_gstd)
        args = args + ' -bandpass '   + ('%f' % self.bandpass_highpass) + (',%f' % self.bandpass_lowpass)
        args = args + ' -rolloff_f '  + ('%f' % self.bandpass_rolloff)
        args = args + ' -symmetry '   + self.symmetry
        if self.rec_halfsets:
            args = args + ' -rec_halves 1'
        else:
            args = args + ' -rec_halves 0'
        return args
    
    def reconstruct(self,out_pfx,tomos_file,ptcls_in,box_size):
        cmd = os.path.dirname(susan.__file__)+'/../bin/susan_reconstruct'
        cmd = cmd + self.get_args(out_pfx,tomos_file,ptcls_in,box_size)
        rslt = os.system(cmd)
        if not rslt == 0:
            raise NameError('Error executing the reconstruction: ' + cmd)
    
    def reconstruct_mpi(self,out_pfx,tomos_file,ptcls_in,box_size,mpi_nodes):
        cmd = 'srun --mpi=pmi2 -n ' + ('%d ' % mpi_nodes)
        cmd = cmd + os.path.dirname(susan.__file__)+'/../bin/susan_reconstruct_mpi'
        cmd = cmd + self.get_args(out_pfx,tomos_file,ptcls_in,box_size)
        rslt = os.system(cmd)
        if not rslt == 0:
            raise NameError('Error executing the reconstruction: ' + cmd)
        
class ref_fsc:
    def __init__(self):
        self.gpu_id    = 0
        self.rand_fpix = 15
        self.pix_size  = -1
        self.threshold = 0.143
        self.save_svg  = False
        
    def calculate(self,out_dir,refs_file):
        cmd = os.path.dirname(susan.__file__)+'/../bin/susan_refs_fsc'
        cmd = cmd + ' -out_dir '   + out_dir
        cmd = cmd + ' -refs_file ' + refs_file
        cmd = cmd + ' -gpu_id '    + ('%d' % self.gpu_id)
        cmd = cmd + ' -rand_fpix ' + ('%f' % self.rand_fpix)
        cmd = cmd + ' -pix_size '  + ('%f' % self.pix_size)
        cmd = cmd + ' -threshold ' + ('%f' % self.threshold)
        if self.save_svg:
            cmd = cmd + ' -save_svg 1'
        else:
            cmd = cmd + ' -save_svg 0'
        rslt = os.system(cmd)
        if not rslt == 0:
            raise NameError('Error calculating the FSCs: ' + cmd)
    
class ref_align:
    def __init__(self):
        self.gpu_id            = 0
        self.bandpass_highpass = 0
        self.bandpass_lowpass  = 20
        self.bandpass_rolloff  = 3
        self.cone_range        = 0
        self.cone_step         = 1
        self.inplane_range     = 0
        self.inplane_step      = 1
        self.refine_level      = 0
        self.refine_factor     = 1
        self.offset_range      = [4,4,4]
        self.offset_step       = 1
        self.offset_type       = 'ellipsoid'
        self.pix_size          = 1
        
    def validate(self):
        if not self.offset_type in ['ellipsoid','cylinider']:
            raise NameError('Invalid offset type. Only "ellipsoid" or "cylinder" are valid')
        
    def align(self,ptcls_out,refs_file,ptcls_in,box_size):
        self.validate()
        cmd = os.path.dirname(susan.__file__)+'/../bin/susan_refs_aligner'
        cmd = cmd + ' -ptcls_in '   + ptcls_in
        cmd = cmd + ' -ptcls_out '  + ptcls_out
        cmd = cmd + ' -refs_file '  + refs_file
        cmd = cmd + ' -gpu_id '     + ('%d' % self.gpu_id)
        cmd = cmd + ' -box_size '   + ('%d' % box_size)
        cmd = cmd + ' -bandpass '   + ('%f' % self.bandpass_highpass) + (',%f' % self.bandpass_lowpass)
        cmd = cmd + ' -rolloff_f '  + ('%f' % self.bandpass_rolloff)
        cmd = cmd + ' -cone '       + ('%f' % self.cone_range) + (',%f' % self.cone_step)
        cmd = cmd + ' -inplane '    + ('%f' % self.inplane_range) + (',%f' % self.inplane_step)
        cmd = cmd + ' -refine '     + ('%f' % self.refine_factor) + (',%f' % self.refine_level)
        cmd = cmd + ' -off_type '   + self.offset_type
        cmd = cmd + ' -off_params ' + ('%f' % self.offset_range[0]) + (',%f' % self.offset_range[1]) + (',%f' % self.offset_range[2]) + (',%f' % self.offset_step)
        cmd = cmd + ' -pix_size '   + ('%f' % self.pix_size)
        rslt = os.system(cmd)
        if not rslt == 0:
            raise NameError('Error calculating the FSCs: ' + cmd)
    
