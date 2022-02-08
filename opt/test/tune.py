import numpy as np
import matplotlib.pyplot as plt
import os
import time
from loguru import logger
from scipy.optimize import minimize
from netCDF4 import Dataset

base_path="/home/tzhang/wrf_solar/wsolar412_bnl/runwrf_tune"
archive_path = "/S2/gscr2/tzhang/big_data/UQ/tune/"
para_names = ['vdis','beta_con']
os.chdir(base_path)

# delete all case
def delete_all_case():
    os.system("rm -rf "+archive_path+"/Iter*")


# check parameter bounds
para_bounds = {
    'vdis':[0.01,1.4],
    'beta_con':[1.02E20, 1.67e24]
}

def check_bounds(x):
    nx = len(x)
    nb = len(para_bounds)
    
    if nx != nb:
        raise SystemExit('shape of input parameters should be consistent with the bounds!')
    
    for key in x:
        x[key] = x[key] if x[key] > para_bounds[key][0] else para_bounds[key][0]
        x[key] = x[key] if x[key] < para_bounds[key][1] else para_bounds[key][1]

    return x


# # calc metrics

def calc_metrics(path_mod):
    var_list = ['dni','dhi'] # dni: direct normal irradiance DHI: diffuse horizontal irradiance
    varn_obs = ['obs_swddni', 'obs_swddif']
    varn_mod = ['SWDDNI','SWDDIF']
    
    path_base = "/ss/hsdc/home/tzhang/wrf_solar/"
    path_obs = path_base+"sgpradflux10long_area_mean.c1.20160619_1200UTC.nc"
    path_def = "/S2/gscr2/tzhang/big_data/UQ/tune/runwrf_def/wrfout_d02_2016-06-19_06:00:00"
    
    fid_obs = Dataset(path_obs)
    fid_def = Dataset(path_def)
    fid_mod = Dataset(path_mod)
    
    Chi = 0
    for i, var in enumerate(var_list):
        #print(var)
        
        var_obs = fid_obs.variables[varn_obs[i]][:]
        var_mod = fid_mod.variables[varn_mod[i]][36:]
        var_def = fid_def.variables[varn_mod[i]][36:]
        
        var_mod_avg = np.mean(var_mod, axis=(1,2))
        var_def_avg = np.mean(var_def, axis=(1,2))
        
        #print(var_obs.shape)
        #print(var_mod_avg.shape)
        #print(var_def_avg.shape)
        
        theta_mod = 0
        for j in range(var_obs.shape[0]):
            theta_mod += (var_obs[j] - var_mod_avg[j]) ** 2
            
        theta_def = 0
        for j in range(var_obs.shape[0]):
            theta_def += (var_obs[j] - var_def_avg[j]) ** 2
            
        Chi += (theta_mod / theta_def)
        
    Chi /= len(var_list)
    
    return Chi       

# # run case

ite = -1
def run_case(x):
    global ite
    ite = ite + 1 
    
    # check parameter bound
    x = check_bounds(x)
    
    # modify namelist
    logger.info("Iter"+str(ite)+": modifiy namelist")
    for key in x:
        replace_str = "sed -i '/^ "+key+"/c\ "+key+"="+str(x[key])+"' namelist.input"
        os.system(replace_str)

    # run model
    logger.info("Iter"+str(ite)+": run model")
    jid = os.popen("qsub runwrf.sh").read().split('.')[0]
    #print(jid)
    while os.popen("qstat").read().find(jid) != -1:
        time.sleep(300)

    mcpi = calc_metrics("/home/tzhang/wrf_solar/wsolar412_bnl/runwrf_tune/wrfout_d02_2016-06-19_06:00:00")

    #archive case
    logger.info("Iter"+str(ite)+": archive case")
    os.system("mkdir -p "+archive_path+"/Iter"+str(ite))
    os.system("cp namelist.input "+archive_path+"/Iter"+str(ite))
    os.system("cp wrfout_* "+archive_path+"/Iter"+str(ite))
    
    logger.info("case"+str(ite)+": success!!!")
    logger.remove()

    return mcpi
    
def run_case_wrapper(x):
    paras = {}
    
    for i,key in enumerate(para_names):
        paras[key] = x[i]
        
    mcpi = run_case(paras)
    return mcpi

initial_simplex = np.array([
    [1.311735, 1.1281180989999998e+24],
    [1.3103449999999999, 4.2008134699999996e+23],
    [0.869715, 1.4955787099999998e+23]
])

x0 = np.array([1.311735, 1.1281180989999998e+24])
res = minimize(run_case_wrapper, x0, method='nelder-mead',
        options={'xatol': 1e-8, 'disp': True, 'return_all':True, 'initial_simplex':initial_simplex})

import pickle

f = open('store.pckl', 'wb')
pickle.dump(res, f)
f.close()
