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

__all__ = ['read_line','read','write']

def _decode_if_needed(line):
    try:
        return line.decode('utf-8')
    except:
        return line

def read_line(fp):
    line = _decode_if_needed( fp.readline().strip() )
    while len(line) > 0 and line[0] == "#" :
        line = _decode_if_needed( fp.readline().strip() )
    return line

def read(fp,tag):
    line = _decode_if_needed( fp.readline().strip() )
    while len(line) > 0 and line[0] == "#" :
        line = _decode_if_needed( fp.readline().strip() )
    if not line.startswith(tag):
        raise NameError("Requested field "+tag+", but the line is "+line)
    return line[(len(tag)+1):]

def write(fp,tag,value):
    fp.write(tag+':'+value+'\n')
