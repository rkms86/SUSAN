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

from dataclasses import dataclass as _dc

@_dc
class fsc_info:
    fpix: float
    res:  float

@_dc
class ssnr:
    S: float
    F: float
    # SSNR(f) = (10^(3*S)) * exp( -f * 100*F )

@_dc
class bandpass:
    highpass: float
    lowpass:  float
    rolloff:  float

@_dc
class search_params:
    span: float
    step: float

@_dc
class offset_params:
    span: list
    step: float
    kind: str

@_dc
class refine_params:
    levels: int
    factor: int

@_dc
class range_params:
    min_val: int
    max_val: int

class mpi_params:
    cmd: str
    arg: int
    
    def __init__(self,cmd=None,arg=None):
        self.cmd = cmd
        self.arg = arg
    
    def gen_cmd(self):
        if self.cmd is None or not self.cmd:
            return ""
        if self.arg is None or not self.arg:
                return self.cmd
        return self.cmd%self.arg + ' '
 
@_dc
class inversion_params:
    ite: int
    std: float
 
@_dc
class boost_lowfreq_params:
    scale: float
    value: float
    decay: float
 

 
 
 
 
 
 
