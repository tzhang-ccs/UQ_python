import numpy as np
import xarray as xr
import scipy.stats
import os
import time
import pandas as pd
import xarray as xr
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('/home/ac.tzhang/E3SM/UQ_python/opt/coupled/TuRBO/')
from turbo import Turbo1

from loguru import logger
logger.remove()
fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <cyan>{level}</cyan> | {message}"
logger.add(sys.stdout, format=fmt)
log_path = f'tuning.log'
if os.path.exists(log_path):
    os.remove(log_path)
ii = logger.add(log_path)

idd = 41
p_names = ['clubb_c1','clubb_gamma_coef','clubb_c_k10','zmconv_tau','zmconv_dmpdz', 'zmconv_micro_dcs', 
            'nucleate_ice_subgrid','p3_nc_autocon_expon','p3_qc_accret_expon','zmconv_auto_fac',
            'zmconv_accr_fac','zmconv_ke','cldfrc_dp1','p3_embryonic_rain_size','effgw_oro']

outputs = ['SWCF global ceres_ebaf_toa_v4.1','LWCF global ceres_ebaf_toa_v4.1','PRECT global GPCP_v2.3',
           'FLNS global ceres_ebaf_surface_v4.1','U-200mb global ERA5','PSL global ERA5','Z3-500mb global ERA5',
           'TREFHT land ERA5','T-200mb global ERA5','SST global HadISST_PI']

output_sh = ['SWCF', 'LWCF', 'PRECT', 'FLNS','U-200mb', 'PSL', 'Z3-500mb', 'TREFHT', 'T-200mb', 'SST']
output_cons = 'RESTOM global ceres_ebaf_toa_v4.1'

def area_mean(data,lat):
    weights = np.cos(np.deg2rad(lat))

    data_weight = data.weighted(weights)
    weighted_mean = data_weight.mean(("lat", "lon"))

    return weighted_mean

def area_std(data, lat):
    weights = np.cos(np.deg2rad(lat))

    data_weight = data.weighted(weights)
    weighted_std = data_weight.std(("lat", "lon"))

    return weighted_std

def get_score(sd_mod, sd_obs, corr):
    score = np.log((((sd_obs/sd_mod + sd_mod/sd_obs) ** 2) * (2 ** 4))/(4 * (1 + corr) ** 4))
    return score

def cost_fun1():
    path_mod = '/lcrc/group/e3sm/ac.tzhang/E3SMv2_1/SCM/e3sm_scm_ARM97/run/'
    path_obs = '/lcrc/group/e3sm/data/inputdata/atm/cam/scam/iop/'

    fid_mod = xr.open_dataset(f'{path_mod}/e3sm_scm_ARM97.eam.h0.1997-06-19-84585.nc')
    fid_obs = xr.open_dataset(f'{path_obs}/ARM97_iopfile_4scam.nc')

    PRECC = fid_mod['PRECC']
    PRECL = fid_mod['PRECL']

    var_mod = (PRECC + PRECL) * 86400 * 1000
    var_obs = fid_obs['Prec'] * 86400

    var_mod_mean = np.mean(var_mod).values
    var_obs_mean = np.mean(var_obs).values

    diff = np.abs(var_mod_mean - var_obs_mean)
    return diff

def cost_fun3():
    #outputs = ['SWCF global ceres_ebaf_toa_v4.1','LWCF global ceres_ebaf_toa_v4.1','PRECT global GPCP_v2.3',
    #       'FLNS global ceres_ebaf_surface_v4.1','U-200mb global ERA5','PSL global ERA5','Z3-500mb global ERA5',
    #       'TREFHT land ERA5','T-200mb global ERA5','SST global HadISST_PI']

    #output_sh = ['SWCF', 'LWCF', 'PRECT', 'FLNS','U-200mb', 'PSL', 'Z3-500mb', 'TREFHT', 'T-200mb', 'SST']
    #output_cons = 'RESTOM global ceres_ebaf_toa_v4.1'

    path_diag = '/home/ac.tzhang/www/e3sm_diags/20230223.NGD_v3atm.piControl.tune/'
    data_diag = pd.read_csv(f'{path_diag}/e3sm_diags/atm_monthly_180x360_aave/model_vs_obs_0001-0010/viewer/table-data/ANN_metrics_table.csv').set_index('Variables')
    path_base = f'/lcrc/group/e3sm/ac.tzhang/E3SMv3/20230223.NGD_v3atm.piControl.tune/'
    path_mpas = "diags/post/analysis/mpas_analysis/ts_0001-0010_climo_0001-0010/timeseries/"

    #tuning varialbs
    data_tune = data_diag.loc[outputs]
    sd_obs = data_tune['Ref._STD'].values
    sd_mod = data_tune['Test_STD'].values
    corr =   data_tune['Correlation'].values

    ts = np.log((((sd_obs/sd_mod + sd_mod/sd_obs) ** 2) * (2 ** 4))/(4 * (1 + corr) ** 4)).sum()

    #secIce area/Volumn
    iceA_def = 11785647.64
    iceA_obs = 9950291.79
    iceV_def = 23.80
    iceV_obs = 20.03

    fid = xr.open_dataset(f'{path_base}/{path_mpas}/seaIceAreaVolNH.nc')
    iceA_mod = np.mean(fid['iceArea'].values / (1000 ** 2))
    iceV_mod = np.mean(fid['iceVolume'].values / (1000 ** 3 * 1000))

    #constraint variables
    constrain = data_diag.loc[output_cons]['Test_mean']
    score = ts + 0.1 * np.abs(iceA_score) + 0.1 * np.abs(iceV_score) + np.abs(constrain)

    path_archive = f"/home/ac.tzhang/fs0_large/E3SM_archive/workdir.{idd}"
    path_climo   = f"/lcrc/group/e3sm/ac.tzhang/E3SMv3/20230223.NGD_v3atm.piControl.tune/diags/post/"
    path_para    = f"/lcrc/group/e3sm/ac.tzhang/E3SMv3/20230223.NGD_v3atm.piControl.tune/"
    os.system(f"mkdir -p {path_archive}/restart")
    os.system(f"cp -r {path_climo} {path_archive}")
    os.system(f"cp {path_para}/run/atm_in {path_para}/case_scripts/user_nl_eam {path_archive}")
    os.system(f"cp -r {path_diag} {path_archive}")
    os.system(f"cp {path_para}/run/rpointer.*  {path_archive}/restart")
    os.system(f"cp {path_para}/run/20230223.NGD_v3atm.piControl.tune.*.{{r,i}}*.0011-01-01*nc  {path_archive}/restart")
    os.system(f"cp {path_para}/run/20230223.NGD_v3atm.piControl.tune.{{eam,elm}}.h0.0010-12*  {path_archive}/restart")
    os.system(f"cp {path_para}/run/20230223.NGD_v3atm.piControl.tune.eam.elm.h1.0010-12*  {path_archive}/restart")
    os.system(f"rm -rf {path_para}/diags/post")
    idd = idd + 1
    return score[0]

def cost_fun2():
    global idd

    #outputs = ['SWCF global ceres_ebaf_toa_v4.1','LWCF global ceres_ebaf_toa_v4.1','PRECT global GPCP_v2.3','FLNS global ceres_ebaf_surface_v4.1',
    #            'U-200mb global ERA5','PSL global ERA5','Z3-500mb global ERA5','TREFHT land ERA5','T-200mb global ERA5','SST global HadISST_PI']
    #output_sh = ['SWCF', 'LWCF', 'PRECT', 'FLNS','U-200mb', 'PSL', 'Z3-500mb', 'TREFHT', 'T-200mb', 'SST']

    path_diag = '/home/ac.tzhang/www/e3sm_diags/20230223.NGD_v3atm.piControl.tune/'
    data_diag = pd.read_csv(f'{path_diag}/e3sm_diags/atm_monthly_180x360_aave/model_vs_obs_0001-0010/viewer/table-data/ANN_metrics_table.csv')
    var_names = data_diag['Variables'].values
    sd_obs = data_diag['Ref._STD'].values
    sd_mod = data_diag['Test_STD'].values
    corr =   data_diag['Correlation'].values
    rmse =   data_diag['RMSE'].values

    ts = np.log((((sd_obs/sd_mod + sd_mod/sd_obs) ** 2) * (2 ** 4))/(4 * (1 + corr) ** 4))

    tmp = {}
    score = 0
    for vn,data in zip(var_names, ts):
        tmp[vn] = data

    for vn in tmp:
        score += tmp[vn]

    #RESTOM, seaice_area, seaice_volume constraint
    lamda1 = 1
    lamda2 = 1
    lamda3 = 1

    restom = data_diag[data_diag.Variables == 'RESTOM global ceres_ebaf_toa_v4.1']['Test_mean'].values
    path_mpas = '/lcrc/group/e3sm/ac.tzhang/E3SMv3/20230223.NGD_v3atm.piControl.tune/diags/post/analysis/mpas_analysis/ts_0001-0010_climo_0001-0010/timeseries/'

    iceA_def = 11785647.64
    iceA_obs = 9950291.79
    iceV_def = 23.80
    iceV_obs = 20.03

    fid = xr.open_dataset(f'{path_mpas}/seaIceAreaVolNH.nc')
    iceA_mod = np.mean(fid['iceArea'].values / (1000 ** 2))
    iceV_mod = np.mean(fid['iceVolume'].values / (1000 ** 3 * 1000))

    iceA_score = (iceA_mod - iceA_obs) / (iceA_def - iceA_obs)
    iceV_score = (iceV_mod - iceV_obs) / (iceV_def - iceV_obs)

    score = score + lamda1 * restom + lamda2 * iceA_score + lamda3 * iceV_score

    path_archive = f"/home/ac.tzhang/fs0_large/E3SM_archive/workdir.{idd}"
    path_climo   = f"/lcrc/group/e3sm/ac.tzhang/E3SMv3/20230223.NGD_v3atm.piControl.tune/diags/post/"
    path_para    = f"/lcrc/group/e3sm/ac.tzhang/E3SMv3/20230223.NGD_v3atm.piControl.tune/"
    os.system(f"mkdir -p {path_archive}/restart")
    os.system(f"cp -r {path_climo} {path_archive}")
    os.system(f"cp {path_para}/run/atm_in {path_para}/case_scripts/user_nl_eam {path_archive}")
    os.system(f"cp -r {path_diag} {path_archive}")
    os.system(f"cp {path_para}/run/rpointer.*  {path_archive}/restart")
    os.system(f"cp {path_para}/run/20230223.NGD_v3atm.piControl.tune.*.{{r,i}}*.0011-01-01*nc  {path_archive}/restart")
    os.system(f"cp {path_para}/run/20230223.NGD_v3atm.piControl.tune.{{eam,elm}}.h0.0010-12*  {path_archive}/restart")
    os.system(f"cp {path_para}/run/20230223.NGD_v3atm.piControl.tune.eam.elm.h1.0010-12*  {path_archive}/restart")
    os.system(f"rm -rf {path_para}/diags/post")
    idd = idd + 1
    return score[0]

def cost_fun3():
    path_mod = '/lcrc/group/e3sm/ac.tzhang/E3SMv2_1/F2010.ne30pg2_EC30to60E2r2_TuRBO_M3/diags/'
    path_obs = '/lcrc/soft/climate/e3sm_diags_data/obs_for_e3sm_diags/climatology/'

    fid_mod = xr.open_dataset(f'{path_mod}/F2010.ne30pg2_EC30to60E2r2_TuRBO_M3_ANN_000101_000212_climo.nc')
    fid1_obs = xr.open_dataset(f'{path_obs}/GPCP_v2.3/GPCP_v2.3_ANN_climo.nc')
    fid2_obs = xr.open_dataset(f'{path_obs}/ceres_ebaf_toa_v4.1/ceres_ebaf_toa_v4.1_ANN_200101_201812_climo.nc')

    var1_mod = (fid_mod['PRECL'] + fid_mod['PRECC']) * 1000 * 24 * 3600
    var2_mod = fid_mod['LWCF']
    lat_mod = fid_mod['lat']
    lon_mod = fid_mod['lon']
    var1_mod_std = area_std(var1_mod,lat_mod)
    var2_mod_std = area_std(var2_mod,lat_mod)
    var1_mod_mean = area_mean(var1_mod,lat_mod)
    var2_mod_mean = area_mean(var2_mod,lat_mod)

    var1_obs = fid1_obs['PRECT']
    var2_obs = fid2_obs['rlutcs'] - fid2_obs['rlut']
    lat1_obs = fid1_obs['lat']
    lat2_obs = fid2_obs['lat']
    lon1_obs = fid1_obs['lon']
    lon2_obs = fid2_obs['lon']
    var1_obs_std = area_std(var1_obs,lat1_obs)
    var2_obs_std = area_std(var2_obs,lat2_obs)
    var1_obs_mean = area_mean(var1_obs,lat1_obs)
    var2_obs_mean = area_mean(var2_obs,lat2_obs)

    var1_mod_new = var1_mod.interp(lat=lat1_obs, lon=lon1_obs)
    var2_mod_new = var2_mod.interp(lat=lat2_obs, lon=lon2_obs)
    
    corr1 = scipy.stats.pearsonr(var1_mod_new.data.reshape(-1), var1_obs.data.reshape(-1))[0]
    corr2 = scipy.stats.pearsonr(var2_mod_new.data.reshape(-1), var2_obs.data.reshape(-1))[0]
    

    score1 = get_score(var1_mod_std.data, var1_obs_std.data, corr1)
    score2 = get_score(var2_mod_std.data, var2_obs_std.data, corr2)

    return score1, score2

def run_model(x):
    path_base = '/home/ac.tzhang/E3SM/UQ_python/opt/coupled'
    path_mod = '/lcrc/group/e3sm/ac.tzhang/E3SMv3/20230223.NGD_v3atm.piControl.tune/case_scripts/'
#    p_names = ['ice_sed_ai', 'clubb_c1','clubb_gamma_coef','zmconv_tau','zmconv_dmpdz']
#    p_names = ['clubb_c1','clubb_gamma_coef','clubb_c_k10','zmconv_tau','zmconv_dmpdz', 'zmconv_micro_dcs', 'nucleate_ice_subgrid','p3_nc_autocon_expon','p3_qc_accret_expon','zmconv_auto_fac','zmconv_accr_fac','zmconv_ke','cldfrc_dp1','p3_embryonic_rain_size','effgw_oro']
    paras = {}
    x = x[0,:]

    mesg = ''
    for i,n in enumerate(p_names):
        paras[n] = x[i]
        mesg = f'{mesg} {n}={x[i]:.5e}'

    os.chdir(path_mod)
    for key in paras:
        replace_str = "sed -i '/\<"+key+"\>/c\ "+key+"="+str(paras[key])+"' user_nl_eam"
        os.system(replace_str)

    os.system("./case.submit >& case_id")
    jid = os.popen("tail -n 1 case_id |awk '{print $6}'").read().strip()
    logger.debug(f'Submit E3SM with job id {jid}')

    while os.popen("squeue -u ac.tzhang").read().find(jid) != -1:
        time.sleep(60)

    os.chdir(path_base)
    os.system("zppy -c post.v2.LR.piControl.cfg >& /dev/null")
    score = cost_fun3()
    mesg = f'{mesg} : score={score:.3f}'
    logger.info(mesg)
    return score


class E3SM:
    def __init__(self, dim=15):
        self.dim = dim
        self.lb = np.array([350,1.0,0.1,1800,-2.0e-3])
        self.lb = np.array([1.0, 0.1, 0.3, 1800, -1e-3,  100e-6, 1,  -1.8, 1.1, 5.0, 1.5, 0.5e-6, 0.01, 15e-6, 0.3])
        #self.ub = np.array([1400,5.0,0.5,14400,-0.1e-3])
        #self.ub = np.array([1400,5.0,0.5,5000,-0.1e-3])
        self.ub = np.array([3.0, 0.3, 0.75,8100, -0.1e-3,250e-6, 1.4,-1.2, 1.3, 7.5, 2.0, 5.0e-6, 0.05, 30e-6, 0.4])
        
    def __call__(self, x):
        x = x.reshape(1,-1)
        val = run_model(x)
        return val

f = E3SM()

turbo1 = Turbo1(
    f=f,  # Handle to objective function
    lb=f.lb,  # Numpy array specifying lower bounds
    ub=f.ub,  # Numpy array specifying upper bounds
    n_init=20,  # Number of initial bounds from an Latin hypercube design
    max_evals = 500,  # Maximum number of evaluations
    batch_size=50,  # How large batch size TuRBO uses
    verbose=True,  # Print information from each batch
    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
    min_cuda=1024,  # Run on the CPU for small datasets
    device="cpu",  # "cpu" or "cuda"
    dtype="float64",  # float64 or float32
)


turbo1.optimize(p_names, outputs, output_cons, sampling_flag=False)
sys.exit()

X = turbo1.X  # Evaluated points
fX = turbo1.fX  # Observed values

np.save('paras',X)
np.save('score',fX)