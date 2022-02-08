

from smt.sampling_methods import LHS
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from loguru import logger
import multiprocessing as mp

para_limits = np.array([
    [0.001,0.01],
    [1.02E20, 1.67e24]
]
)

num = 200

base_path="/home/tzhang/wsolar412_bnl_v2/"
archive_path = "/backup2/tzhang/uq/sampling/"
para_names = ['dis_alpha','beta_con']

sampling = LHS(xlimits=para_limits)
x = sampling(num)
print(x.shape)
print(x[0,:])

para_samples = []
for id, xx in enumerate(x):
    samp = {}
    for i,name in enumerate(para_names):
        samp[name] = xx[i]
    para_samples.append((id,samp))
    

def run_case(id,x):
    #create a case
    #print("cp -r "+base_path+"/runwrf "+base_path+"/case"+str(id))
    archive_case = archive_path+"/case"+str(id)
    os.system("mkdir -p "+archive_case)

    os.system("cp -r "+base_path+"/runwrf "+base_path+"/case"+str(id))
    os.chdir(base_path+"/case"+str(id))
    logger.add(archive_case+"/case"+str(id)+".log")
    logger.info("case"+str(id)+": create a case")
    logger.info("case"+str(id)+": "+str(x))
    #os.system("touch a")
    
    # modify runwrf.sh
    logger.info("case"+str(id)+": modifiy runwrf.sh")
    os.system("sed -i 's/runwrf/case"+str(id)+"/g' runwrf.sh")
    
    # modify namelist
    logger.info("case"+str(id)+": modifiy namelist")
    for key in x:
        replace_str = "sed -i '/^ "+key+"/c\ "+key+"="+str(x[key])+"' namelist.input"
        os.system(replace_str)
    
    # run model
    logger.info("case"+str(id)+": run model")
    #time.sleep(30)
    jid = os.popen("qsub runwrf.sh").read().split('.')[0]
    #print(jid)
    while os.popen("qstat").read().find(jid) != -1:
        time.sleep(300)
        
    #archive case
    logger.info("case"+str(id)+": archive case")
    os.system("cp wrfout_d02_2016-06-19_06:00:00 namelist.input "+archive_case)
    os.chdir(base_path)
    os.system("rm -rf "+"case"+str(id))
    
    logger.info("case"+str(id)+": success!!!")
    logger.remove()
    
def run_case_wrapper(args):
    return run_case(*args)

pool = mp.Pool(20)
pool.map(run_case_wrapper,para_samples)
