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

from . import data
from . import io
from . import utils
from . import modules
from . import project

def read(filename):
	if utils.is_extension(filename,'mrc') or  utils.is_extension(filename,'map') or utils.is_extension(filename,'ali') or utils.is_extension(filename,'st'):
		v,_ = io.mrc.read(filename)
		return v
	elif utils.is_extension(filename,'ptclsraw'):
		return data.Particles(filename)
	elif utils.is_extension(filename,'refstxt'):
		return data.Reference(filename)
	elif utils.is_extension(filename,'tomostxt'):
		return data.Tomograms(filename)
	else:
		raise ValueError('Unsupported file.')

__all__ = []
__all__.extend(['data','io','utils','modules','project','read'])
