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
import struct, typing

class Particles:
    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            if isinstance(args[0], str):
                filename = args[0] 
                self.load(filename)
            #elif isinstance(args[0],np.ndarray)
            #    # Parse positions....
            else:
                raise NameError('Invalid input')
        else:
            if len(args) == 3:
                n_ptcl = args[0]
                n_proj = args[1]
                n_refs = args[2]
                self.alloc(n_ptcl,n_proj,n_refs)
            else:
                raise NameError('Invalid input')

    def load(self,filename):
        fp = open(filename,"rb")
        buffer = fp.read(8)
        signature = buffer.decode("utf-8")
        if signature != "SsaPtcl1" :
            raise NameError("Invalid signature")
        buffer = fp.read(4*3)
        self.n_ptcl, self.n_proj, self.n_refs = struct.unpack('III',buffer)
        self.__alloc_empty__()
        n_bytes_per_ptcl = 4*( 10 + 8*self.n_refs + 7*self.n_proj + 8*self.n_proj )
        for index in range(self.n_ptcl):
            buffer = fp.read(n_bytes_per_ptcl)
            self.__parse_buffer__(index,buffer)
        fp.close()
    
    def alloc(self,n_ptcl,n_proj,n_refs):
        self.n_ptcl = n_ptcl
        self.n_proj = n_proj
        self.n_refs = n_refs
        self.__alloc_empty__()
    
    def save(self,filename):
        fp = open(filename,"wb")
        signature = "SsaPtcl1"
        fp.write( bytearray(signature,'utf-8') )
        fp.write( struct.pack('III',self.n_ptcl,self.n_proj,self.n_refs) )
        for index in range(self.n_ptcl):
            fp.write( struct.pack('III',self.ptcl_id[index],self.tomo_id[index],self.tomo_cix[index]) )
            fp.write( struct.pack('fff',self.position[index,0],self.position[index,1],self.position[index,2]) )
            fp.write( struct.pack('II' ,int(self.ref_cix[index]),int(self.half_id[index]) ) )
            fp.write( struct.pack('ff' ,self.extra_1[index],self.extra_2[index] ) )
            
            # 3D alignment
            for j in range(self.n_refs):
                fp.write( struct.pack('fff' ,self.ali_eu[index,0,j],self.ali_eu[index,1,j],self.ali_eu[index,2,j] ) )
            for j in range(self.n_refs):
                fp.write( struct.pack('fff' ,self.ali_t[index,0,j],self.ali_t[index,1,j],self.ali_t[index,2,j] ) )
            for j in range(self.n_refs):
                fp.write( struct.pack('f' ,self.ali_cc[index,0,j] ) )
            for j in range(self.n_refs):
                fp.write( struct.pack('f' ,self.ali_w[index,0,j] ) )
            
            # 2D alignment
            for j in range(self.n_proj):
                fp.write( struct.pack('fff' ,self.prj_eu[index,0,j],self.prj_eu[index,1,j],self.prj_eu[index,2,j] ) )
            for j in range(self.n_proj):
                fp.write( struct.pack('ff' ,self.prj_t[index,0,j],self.prj_t[index,1,j] ) )
            for j in range(self.n_proj):
                fp.write( struct.pack('f' ,self.prj_cc[index,0,j] ) )
            for j in range(self.n_proj):
                fp.write( struct.pack('f' ,self.prj_w[index,0,j] ) )
            
            # Defocus
            for j in range(self.n_proj):
                fp.write( struct.pack('fff' ,self.def_U[index,j],self.def_V[index,j],self.def_ang[index,j] ) )
                fp.write( struct.pack('fff' ,self.def_phas[index,j],self.def_Bfct[index,j],self.def_ExFl[index,j] ) )
                fp.write( struct.pack('ff'  ,self.def_mres[index,j],self.def_scor[index,j] ) )
        fp.close()
    
    def __getitem__(self,idx):
        tmp = self.ptcl_id[idx]
        ptcls_out = particles(tmp.shape[0],self.n_proj,self.n_refs)
        ptcls_out.ptcl_id  = self.ptcl_id [idx]
        ptcls_out.tomo_id  = self.tomo_id [idx]
        ptcls_out.tomo_cix = self.tomo_cix[idx]
        ptcls_out.position = self.position[idx,:]
        ptcls_out.ref_cix  = self.ref_cix[idx]
        ptcls_out.half_id  = self.half_id[idx]
        ptcls_out.extra_1  = self.extra_1[idx]
        ptcls_out.extra_2  = self.extra_2[idx]
        # 3D alignment
        ptcls_out.ali_eu   = self.ali_eu[idx,:,:]
        ptcls_out.ali_t    = self.ali_t [idx,:,:]
        ptcls_out.ali_cc   = self.ali_cc[idx,:,:]
        ptcls_out.ali_w    = self.ali_w [idx,:,:]
        # 2D alignment
        ptcls_out.prj_eu   = self.prj_eu[idx,:,:]
        ptcls_out.prj_t    = self.prj_t [idx,:,:]
        ptcls_out.prj_cc   = self.prj_cc[idx,:,:]
        ptcls_out.prj_w    = self.prj_w [idx,:,:]
        # Defocus
        ptcls_out.def_U    = self.def_U   [idx,:]
        ptcls_out.def_V    = self.def_V   [idx,:]
        ptcls_out.def_ang  = self.def_ang [idx,:]
        ptcls_out.def_phas = self.def_phas[idx,:]
        ptcls_out.def_Bfct = self.def_Bfct[idx,:]
        ptcls_out.def_ExFl = self.def_ExFl[idx,:]
        ptcls_out.def_mres = self.def_mres[idx,:]
        ptcls_out.def_scor = self.def_scor[idx,:]
        return ptcls_out
    
    def select(self,idx):
        w_idx = idx[:,0]
        tmp = self.ptcl_id[w_idx]
        ptcls_out = particles(tmp.shape[0],self.n_proj,self.n_refs)
        ptcls_out.ptcl_id  = self.ptcl_id [w_idx]
        ptcls_out.tomo_id  = self.tomo_id [w_idx]
        ptcls_out.tomo_cix = self.tomo_cix[w_idx]
        ptcls_out.position = self.position[w_idx,:]
        ptcls_out.ref_cix  = self.ref_cix[w_idx]
        ptcls_out.half_id  = self.half_id[w_idx]
        ptcls_out.extra_1  = self.extra_1[w_idx]
        ptcls_out.extra_2  = self.extra_2[w_idx]
        # 3D alignment
        ptcls_out.ali_eu   = self.ali_eu[w_idx,:,:]
        ptcls_out.ali_t    = self.ali_t [w_idx,:,:]
        ptcls_out.ali_cc   = self.ali_cc[w_idx,:,:]
        ptcls_out.ali_w    = self.ali_w [w_idx,:,:]
        # 2D alignment
        ptcls_out.prj_eu   = self.prj_eu[w_idx,:,:]
        ptcls_out.prj_t    = self.prj_t [w_idx,:,:]
        ptcls_out.prj_cc   = self.prj_cc[w_idx,:,:]
        ptcls_out.prj_w    = self.prj_w [w_idx,:,:]
        # Defocus
        ptcls_out.def_U    = self.def_U   [w_idx,:]
        ptcls_out.def_V    = self.def_V   [w_idx,:]
        ptcls_out.def_ang  = self.def_ang [w_idx,:]
        ptcls_out.def_phas = self.def_phas[w_idx,:]
        ptcls_out.def_Bfct = self.def_Bfct[w_idx,:]
        ptcls_out.def_ExFl = self.def_ExFl[w_idx,:]
        ptcls_out.def_mres = self.def_mres[w_idx,:]
        ptcls_out.def_scor = self.def_scor[w_idx,:]
        return ptcls_out
    
    def __alloc_empty__(self):
        self.ptcl_id  = np.zeros(self.n_ptcl,dtype=np.uint32)
        self.tomo_id  = np.zeros(self.n_ptcl,dtype=np.uint32)
        self.tomo_cix = np.zeros(self.n_ptcl,dtype=np.uint32)
        self.position = np.zeros((self.n_ptcl,3),dtype=np.float32) # in Angstroms
        self.ref_cix  = np.zeros(self.n_ptcl,dtype=np.uint32)
        self.half_id  = np.zeros(self.n_ptcl,dtype=np.uint32)
        self.extra_1  = np.zeros(self.n_ptcl,dtype=np.float32)
        self.extra_2  = np.zeros(self.n_ptcl,dtype=np.float32)
        
        # 3D alignment
        self.ali_eu   = np.zeros((self.n_ptcl,3,self.n_refs),dtype=np.float32) # in Radians
        self.ali_t    = np.zeros((self.n_ptcl,3,self.n_refs),dtype=np.float32) # in Angstroms
        self.ali_cc   = np.zeros((self.n_ptcl,1,self.n_refs),dtype=np.float32)
        self.ali_w    = np.zeros((self.n_ptcl,1,self.n_refs),dtype=np.float32)
        
        # 2D alignment
        self.prj_eu   = np.zeros((self.n_ptcl,3,self.n_proj),dtype=np.float32) # in Radians
        self.prj_t    = np.zeros((self.n_ptcl,2,self.n_proj),dtype=np.float32) # in Angstroms
        self.prj_cc   = np.zeros((self.n_ptcl,1,self.n_proj),dtype=np.float32)
        self.prj_w    = np.zeros((self.n_ptcl,1,self.n_proj),dtype=np.float32)
        
        # Defocus
        self.def_U    = np.zeros((self.n_ptcl,self.n_proj),dtype=np.float32) # U (angstroms)
        self.def_V    = np.zeros((self.n_ptcl,self.n_proj),dtype=np.float32) # V (angstroms)
        self.def_ang  = np.zeros((self.n_ptcl,self.n_proj),dtype=np.float32) # angles (sexagesimal)
        self.def_phas = np.zeros((self.n_ptcl,self.n_proj),dtype=np.float32) # phase shift (sexagesimal?)
        self.def_Bfct = np.zeros((self.n_ptcl,self.n_proj),dtype=np.float32) # Bfactor
        self.def_ExFl = np.zeros((self.n_ptcl,self.n_proj),dtype=np.float32) # Exposure filter
        self.def_mres = np.zeros((self.n_ptcl,self.n_proj),dtype=np.float32) # Max. resolution (angstroms)
        self.def_scor = np.zeros((self.n_ptcl,self.n_proj),dtype=np.float32) # score
        
    def __parse_buffer__(self,index,buffer):
        i=0
        self.ptcl_id [index], self.tomo_id [index], self.tomo_cix[index], \
        self.position[index,0], self.position[index,1], self.position[index,2], \
        self.ref_cix [index], self.half_id [index], \
        self.extra_1 [index], self.extra_2 [index] =  struct.unpack('IIIfffIIff',buffer[i:i+40])
        i=i+40
        
        # 3D alignment
        for j in range(self.n_refs):
            self.ali_eu[index,0:3,j] = struct.unpack('fff',buffer[i:i+12])
            i=i+12
        for j in range(self.n_refs):
            self.ali_t [index,0:3,j] = struct.unpack('fff',buffer[i:i+12])
            i=i+12
        for j in range(self.n_refs):
            self.ali_cc[index,0,j] = struct.unpack('f',buffer[i:i+4])[0]
            i=i+4;
        for j in range(self.n_refs):
            self.ali_w [index,0,j] = struct.unpack('f',buffer[i:i+4])[0]
            i=i+4;
        
        # 2D alignment
        for j in range(self.n_proj):
            self.prj_eu[index,0:3,j] = struct.unpack('fff',buffer[i:i+12])
            i=i+12
        for j in range(self.n_proj):
            self.prj_t [index,0:2,j] = struct.unpack('ff',buffer[i:i+8])
            i=i+8
        for j in range(self.n_proj):
            self.prj_cc[index,0,j] = struct.unpack('f',buffer[i:i+4])[0]
            i=i+4
        for j in range(self.n_proj):
            self.prj_w [index,0,j] = struct.unpack('f',buffer[i:i+4])[0]
            i=i+4
        
        # Defocus
        for j in range(self.n_proj):
            self.def_U[index,j], self.def_V[index,j], self.def_ang[index,j], \
            self.def_phas[index,j], self.def_Bfct[index,j], self.def_ExFl[index,j], \
            self.def_mres[index,j], self.def_scor[index,j] = struct.unpack('ffffffff',buffer[i:i+32])
            i=i+32

def _decode_if_needed(line):
    try:
        return line.decode('utf-8')
    except:
        return line

def read_value(fp,tag):
    line = _decode_if_needed( fp.readline().strip() )
    while len(line) > 0 and line[0] == "#" :
        line = _decode_if_needed( fp.readline().strip() )
    if not line.startswith(tag):
        raise NameError("Requested field "+tag+", but the line is "+line)
    return line[(len(tag)+1):]

def write_value(fp,tag,value):
    fp.write(tag+':'+value+'\n')
    
class reference(typing.NamedTuple):
    ref  : str
    mask : str
    h1   : str
    h2   : str

def load_references(filename):
    fp = open(filename,"rb")
    num_refs = int(read_value(fp,'num_ref'))
    names=[]
    masks=[]
    half1=[]
    half2=[]
    for i in range(num_refs):
        names.append(read_value(fp,'map'))
        masks.append(read_value(fp,'mask'))
        half1.append(read_value(fp,'h1'))
        half2.append(read_value(fp,'h2'))
    fp.close()
    result = [
        reference(*refs_details)
        for refs_details in zip(names, masks, half1, half2)
    ]
    return result

def save_references(refs,filename):
    fp=open(filename,'w')
    write_value(fp,'num_ref',str(len(refs)))
    for i in range(len(refs)):
        fp.write('## Reference '+str(i+1)+'\n')
        write_value(fp,'map',refs[i].ref)
        write_value(fp,'mask',refs[i].mask)
        write_value(fp,'h1',refs[i].h1)
        write_value(fp,'h2',refs[i].h2)
    fp.close()

