

from smt.sampling_methods import LHS
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from loguru import logger
import multiprocessing as mp

para_limits = np.array([
    [0.01,1.4],
    [1.02E20, 1.67e24]
]
)

num = 1000

base_path="/home/tzhang/wrf_solar/wsolar412_bnl/"
archive_path = "/S2/gscr2/tzhang/big_data/UQ/sampling/"
para_names = ['vdis','beta_con']

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
    os.system("cp -r "+base_path+"/runwrf "+base_path+"/case"+str(id))
    os.chdir(base_path+"/case"+str(id))
    logger.add("case"+str(id)+".log")
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
    os.chdir(base_path)
    #print("mv case"+str(id)+" "+archive_path)
    os.system("mv case"+str(id)+" "+archive_path) 
    
    logger.info("case"+str(id)+": success!!!")
    logger.remove()
    
def run_case_wrapper(args):
    return run_case(*args)

pool = mp.Pool(30)
pool.map(run_case_wrapper,para_samples)





