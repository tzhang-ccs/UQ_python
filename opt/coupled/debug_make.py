import pandas as pd
import numpy as np
import xarray as xr
import os

idd = 7

def cost_fun2():
    global idd

    outputs = ['SWCF global ceres_ebaf_toa_v4.1','LWCF global ceres_ebaf_toa_v4.1','PRECT global GPCP_v2.3','FLNS global ceres_ebaf_surface_v4.1',
                'U-200mb global ERA5','PSL global ERA5','Z3-500mb global ERA5','TREFHT global ERA5','T-200mb global ERA5','SST global HadISST_PI']
    output_sh = ['SWCF', 'LWCF', 'PRECT', 'FLNS','U-200mb', 'PSL', 'Z3-500mb', 'TREFHT', 'T-200mb', 'SST']

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
    os.system(f"mkdir -p {path_archive}")
    os.system(f"cp -r {path_climo} {path_archive}")
    os.system(f"cp {path_para}/run/atm_in {path_para}/case_scripts/user_nl_eam {path_archive}")
    os.system(f"cp -r {path_diag} {path_archive}")
    os.system(f"cp {path_para}/run/20230223.NGD_v3atm.piControl.tune.eam.i.0011-01-01-00000.nc {path_archive}")
    os.system(f"cp {path_para}/run/20230223.NGD_v3atm.piControl.tune.{{eam,elm,cpl,mosart,mpaso,mpassi}}.r*.0011-01-01*00000.nc {path_archive}")
#    os.system(f"rm -rf {path_para}/diags/post")
    idd = idd + 1
    return score[0]

score = cost_fun2()
print(score)
print(type(score))
