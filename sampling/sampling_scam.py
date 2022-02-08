

from smt.sampling_methods import LHS
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from loguru import logger
import multiprocessing as mp

para_limits = np.array([
    [2.95e-3, 8.85e-3]
    #[8.0e-1, 9.5e-1]
]
)

num = 1000

base_path="/ss/hsdc/home/tzhang/scam_causal/SCAM_cesm/cesm1_2_2/EXP_causal/"
archive_path = "/S2/gscr2/tzhang/big_data/UQ/scam_sampling/1p"
para_names = ['zmconv_c0_lnd']
#para_names = ['zmconv_c0_lnd','cldfrc_rhminl']

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
    os.chdir(base_path)
    os.system("cp -r run case"+str(id))
    logger.info("case"+str(id)+": create a case")
    
    # modify namelist
    os.chdir(base_path+"/case"+str(id))
    for key in x:
        #replace_str = "sed -i '/^ "+key+"/c\ "+key+"="+str(x[key])+"' atm_in"
        replace_str = "sed -i '/\<"+key+"\>/c\ "+key+"="+str(x[key])+"' atm_in"
        os.system(replace_str)
    
    # run model
    #time.sleep(30)
    os.system("./scam >& /dev/null")
        
    #archive case
    archive_case = archive_path+"/case"+str(id)
    os.system("mkdir -p "+archive_case)
    os.system("cp atm_in "+archive_case)
    os.system("cp camrun.cam.h1.0095-07-19-00000.nc "+archive_case)
    os.system("rm -rf ../case"+str(id))
    
def run_case_wrapper(args):
    return run_case(*args)

pool = mp.Pool(20)
pool.map(run_case_wrapper,para_samples)
