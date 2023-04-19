from bayes_opt import BayesianOptimization
import os
import pandas as pd
import sys
import xarray as xr
import matplotlib.pyplot as plt
from loguru import logger
from datetime import datetime
import time
#from pathlib import Path
#home = str(Path.home())
#sys.path.append(home+"/UQ_python/metrics/")
#import metrics
from sklearn.metrics import mean_squared_error
#from xgboost import XGBRegressor
import numpy as np
from scipy.optimize import minimize

now = datetime.now()
dt_string = now.strftime("%Y_%m_%d-%H_%M_%S")
logger.remove()
fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <cyan>{level}</cyan> | {message}"
logger.add(sys.stdout, format=fmt)
logger.add("logs/log_tune_"+dt_string,format=fmt)

varlist_mod = ['PRECT']
path_mod = "/global/cscratch1/sd/zhangtao/E3SM/SCM_runs/e3sm_scm_ARM97_tune/run/"
base_mod = "/global/cscratch1/sd/zhangtao/E3SM/SCM_runs/e3sm_scm_ARM97_tune/case_scripts/"

varlist_obs = ['Prec']
path_obs = "/global/cfs/cdirs/e3sm/inputdata/atm/cam/scam/iop/"

iter = 1

def scam_costfun():
    global iter

    fid_mod = xr.open_dataset(path_mod+"e3sm_scm_ARM97_tune.eam.h1.1997-06-19-84585.nc")
    for var in varlist_mod:
        var_mod = fid_mod[var].values

        if var == 'PRECT':
            var_mod = var_mod * 1000 * 86400
    mean_mod = np.mean(var_mod)

    fid_obs = xr.open_dataset(path_obs+"ARM97_iopfile_4scam.nc")
    for var in varlist_obs:
        var_obs = fid_obs[var][:,0,0].values

        if var == 'Prec':
            var_obs = var_obs * 86400
    mean_obs = np.mean(var_obs)

    #backup files
    os.chdir(path_mod)
    os.system("mkdir -p Iter"+str(iter))
    os.system("mv e3sm_scm_ARM97_tune.eam.h1.1997-06-19-84585.nc Iter"+str(iter))
    os.system("mv atm_in Iter"+str(iter))
    iter = iter + 1

    return np.abs(mean_mod-mean_obs)

def scam_run(zmconv_c0_ocn, zmconv_dmpdz):
    x = {}
    x['zmconv_c0_ocn'] = zmconv_c0_ocn 
    x['zmconv_dmpdz'] = zmconv_dmpdz

    logger.info("Parameter values are:")
    for key in x:
        logger.info(f'\t{key}:\t {x[key]}')
    
    os.chdir(base_mod)

    for key in x:
        replace_str = "sed -i '/\<"+key+"\>/c\ "+key+"="+str(x[key])+"' user_nl_eam"
        os.system(replace_str)
    
    os.system("./case.submit > case_id")
    jid = os.popen("tail -n 1 case_id |awk '{print $6}'").read().strip()
    logger.info(f'Submit SCAM with job id {jid}')
    
    while os.popen("squeue -u zhangtao").read().find(jid) != -1:
        time.sleep(300)

    y = scam_costfun()
    logger.info(f'Cost metrics is: {y:.2f}')

    return -y

def test_func(zmconv_c0_ocn, zmconv_dmpdz):
    aa = np.random.rand(1,1)
    return aa[0,0]

param_Nlist = ['zmconv_c0_ocn', 'zmconv_dmpdz']
param_bounds = [(0.001, 0.01), (-0.9e-3, -0.4e-3)]

logger.info("Parameter bounds are:")
pbounds = {}
for i,p in enumerate(param_Nlist):
    logger.info(f'\t{p}:\t{param_bounds[i]}')
    pbounds[p] = param_bounds[i]


Boptimizer = BayesianOptimization(
    f=scam_run,
    #f=test_func,
    pbounds=pbounds,
    random_state=10,
)

Boptimizer.maximize(
    init_points=20,
    n_iter=200,
)

#param_dict = {'zmconv_c0_ocn':'0.002', 'zmconv_dmpdz':'-0.7e-3'}
#logger.info("Parameter values are:")
#for key in param_dict:
#    logger.info(f'\t{key}:\t {param_dict[key]}')

#y = scam_run(param_dict)
