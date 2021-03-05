import os,susan,math,os.path,collections,datetime,numpy
import susan.data
import susan.modules

class iteration_files:
    def __init__(self):
        self.ptcl_rslt = ''
        self.ptcl_temp = ''
        self.reference = ''
        self.ite_dir   = ''
        
    def check(self):
        if not os.path.exists(self.ptcl_rslt):
            raise NameError('File '+ self.ptcl_rslt + ' does not exist')
        if not os.path.exists(self.reference):
            raise NameError('File '+ self.reference + ' does not exist')

class manager:
    def __init__(self,prj_name,box_size=-1):
        if box_size < 0:
            if os.path.exists(prj_name):
                if os.path.isdir(prj_name):
                    fp = open(prj_name+"/info.prjtxt","r")
                    self.prj_name = susan.data.read_value(fp,"name")
                    self.box_size = int(susan.data.read_value(fp,"box_size"))
                    fp.close()
                else:
                    raise NameError('Project does not exist')
            else:
                raise NameError('Project does not exist')
        else:
            if not os.path.exists(prj_name):
                os.mkdir(prj_name)
            fp = open(prj_name+"/info.prjtxt","w")
            susan.data.write_value(fp,'name',prj_name)
            susan.data.write_value(fp,'box_size',str(box_size))
            fp.close()
            self.prj_name = prj_name
            self.box_size = box_size
        
        self.mpi_nodes         = 1
        self.list_gpus_ids     = [0]
        self.threads_per_gpu   = 1
        self.dimensionality    = 3
        self.halfsets_independ = False
        self.tomogram_file     = ''
        self.initial_reference = ''
        self.initial_particles = ''
        self.cc_threshold      = 0.8
        self.fsc_threshold     = 0.143
        self.aligner           = susan.modules.aligner()
        self.averager          = susan.modules.averager()
        self.ref_aligner       = susan.modules.ref_align()
        self.ref_fsc           = susan.modules.ref_fsc()
        
        self.aligner.ctf_correction = 'on_reference'
        
        self.averager.ctf_correction    = 'wiener'
        self.averager.rec_halfsets      = True
        self.averager.bandpass_highpass = 0
        self.averager.bandpass_lowpass  = (self.box_size)/2
        
        self.ref_aligner.cone_range    = 2
        self.ref_aligner.cone_step     = 0.5
        self.ref_aligner.inplane_range = 2
        self.ref_aligner.inplane_step  = 0.5
        self.ref_aligner.refine_level  = 2
        self.ref_aligner.refine_factor = 2
    
    def get_iteration_dir(self,ite):
        return self.prj_name + '/ite_' + ('%04d' % ite)
    
    def get_iteration_files(self,ite):
        rslt = iteration_files()
        if ite < 1:
            rslt.ptcl_rslt = self.initial_particles
            rslt.reference = self.initial_reference
        else:
            base_dir = self.get_iteration_dir(ite)
            rslt.ptcl_rslt = base_dir + '/particles.ptclsraw'
            rslt.ptcl_temp = base_dir + '/temp.ptclsraw'
            rslt.reference = base_dir + '/reference.refstxt'
            rslt.ite_dir   = base_dir
        return rslt

    def setup_iteration(self,ite):
        base_dir = self.get_iteration_dir(ite)
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        cur = self.get_iteration_files(ite)
        prv = self.get_iteration_files(ite-1)
        prv.check()
        return (cur,prv)

    def exec_alignment(self,cur,prv):
        self.aligner.list_gpus_ids     = self.list_gpus_ids
        self.aligner.threads_per_gpu   = self.threads_per_gpu
        self.aligner.dimensionality    = self.dimensionality
        self.aligner.halfsets_independ = self.halfsets_independ
        
        if self.aligner.dimensionality == 3:
            header = '  [3D Alignment] '
        else:
            if self.aligner.dimensionality == 2:
                header = '  [2D Alignment] '
            else:
                raise NameError('Invalid dimensionality in the project')
        
        print( header + 'Start:' )
        start_time = datetime.datetime.now()
        if self.mpi_nodes > 1:
            self.aligner.align_mpi(cur.ptcl_rslt,prv.reference,self.tomogram_file,prv.ptcl_rslt,self.box_size,self.mpi_nodes)
        else:
            self.aligner.align(cur.ptcl_rslt,prv.reference,self.tomogram_file,prv.ptcl_rslt,self.box_size)
        end_time = datetime.datetime.now()
        elapsed = end_time-start_time
        print( header + 'Finished using ' + ('%.1f' % elapsed.total_seconds()) + ' seconds (' + str(elapsed) +  ').'  )
        
    def exec_particle_selection(self,cur):
        print('  [Aligned partices] Processing')
        ptcls_in = susan.data.particles(cur.ptcl_rslt)
        
        # Classify
        if ptcls_in.n_refs > 1 :
            ptcls_in.ref_cix = numpy.argmax(ptcls_in.ali_cc,axis=2)
            ptcls_in.save(cur.ptcl_rslt)
        
        # Select particles for reconstruction
        for i in range(ptcls_in.n_refs):
            idx = ptcls_in.ref_cix == i
            hid = ptcls_in.half_id[idx]
            ccc = ptcls_in.ali_cc [idx,:,i].flatten()
            
            th1 = numpy.quantile(ccc[ hid==1 ], 1-self.cc_threshold)
            th2 = numpy.quantile(ccc[ hid==2 ], 1-self.cc_threshold)            
            
            hid[ numpy.logical_and(hid==1,ccc<th1) ] = 0
            hid[ numpy.logical_and(hid==2,ccc<th2) ] = 0
            
            print('    Class ' + str(i+1) + ': ' + str( len(idx) ) + ' particles.' )
            print('      Half 1: ' + str( sum( hid==1 ) ) + ' particles.' )
            print('      Half 2: ' + str( sum( hid==2 ) ) + ' particles.' )
            
        ptcls_out = ptcls_in[ hid>0 ]
        ptcls_out.save(cur.ptcl_temp)
        print('  [Aligned partices] Done.')
        
    def exec_averaging(self,cur,prv):
        self.averager.list_gpus_ids     = self.list_gpus_ids
        self.averager.threads_per_gpu   = self.threads_per_gpu
        
        print( '  [Reconstruct Maps] Start:' )
        start_time = datetime.datetime.now()
        if self.mpi_nodes > 1:
            self.averager.reconstruct_mpi(cur.ite_dir+'/map',self.tomogram_file,cur.ptcl_temp,self.box_size,self.mpi_nodes)
        else:
            self.averager.reconstruct(cur.ite_dir+'/map',self.tomogram_file,cur.ptcl_temp,self.box_size)
        end_time = datetime.datetime.now()
        elapsed = end_time-start_time
        print( '  [Reconstruct Maps] Finished using ' + ('%.1f' % elapsed.total_seconds()) + ' seconds (' + str(elapsed) +  ').'  )
        os.remove(cur.ptcl_temp)
        
        refs_in = susan.data.load_references(prv.reference)
        names=[]
        masks=[]
        half1=[]
        half2=[]
        for i in range(len(refs_in)):
            names.append( cur.ite_dir + '/map_class' + ('%03d'%(i+1)) + '.mrc' )
            half1.append( cur.ite_dir + '/map_class' + ('%03d'%(i+1)) + '_half1.mrc' )
            half2.append( cur.ite_dir + '/map_class' + ('%03d'%(i+1)) + '_half2.mrc' )
            masks.append( refs_in[i].mask )
        refs_out = [
            susan.data.reference(*refs_details)
            for refs_details in zip(names, masks, half1, half2)
        ]
        susan.data.save_references(refs_out,cur.reference)
    
    def exec_fsc(self,cur):
        self.ref_fsc.gpu_id = self.list_gpus_ids[0]
        print( '  [FSC Calculation] Start:' )
        self.ref_fsc.calculate(cur.ite_dir+'/',cur.reference)
        print( '  [FSC Calculation] Done.' )
        
        min_fp = self.box_size/2
        fp = open(cur.ite_dir+'/resolution_result.txt',"r")
        num_refs = int(susan.data.read_value(fp,'num_ref'))
        for i in range(num_refs):
            cur_fp = float( susan.data.read_value(fp,'max_fpix') )
            min_fp = min(min_fp,cur_fp)
        fp.close()
        return min_fp
    
    def execute_iteration(self,ite):
        cur,prv = self.setup_iteration(ite)
        self.exec_alignment(cur,prv)
        self.exec_particle_selection(cur)
        self.exec_averaging(cur,prv)
        rslt = self.exec_fsc(cur)
        return rslt
