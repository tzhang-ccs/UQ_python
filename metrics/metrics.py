from netCDF4 import Dataset
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
from netCDF4 import Dataset

path_base = "/home/zhangtao/scratch/wrf-solar/"

def chi_metrics(path_mod):

    # define path and load data
    var_list = ['dir','dif'] # dni: direct normal irradiance DHI: diffuse horizontal irradiance
    varn_obs = ['obs_swddir', 'obs_swddif']
    varn_mod = ['SWDDIR','SWDDIF']

    path_obs = path_base+"sgpradflux10long_area_mean.c1.20160619_1200UTC_0418.nc"
    path_def = path_base+"wrfout_d02_2016-06-19_06_00_00"

    fid_obs = Dataset(path_obs)
    fid_def = Dataset(path_def)
    fid_mod = Dataset(path_mod)

    data_obs = {}
    data_def = {}
    data_mod = {}

    for i,varn in enumerate(var_list):
        data_obs[varn] = fid_obs.variables[varn_obs[i]][:]
        data_mod[varn] = np.mean(fid_mod.variables[varn_mod[i]][36:],axis=(1,2))
        data_def[varn] = np.mean(fid_def.variables[varn_mod[i]][36:],axis=(1,2))

    data_obs['tot'] = data_obs['dir'] + data_obs['dif']
    data_mod['tot'] = data_mod['dir'] + data_mod['dif']
    data_def['tot'] = data_def['dir'] + data_def['dif']


    #regime classification
    F_all = fid_obs.variables['obs_swdtot'][:]
    F_clr = fid_obs.variables['obs_swdtotc'][:]
    F_all_drt = fid_obs.variables['obs_swddir'][:]
    F_clr_drt = fid_obs.variables['obs_swddirc'][:]
    eps = 10**-6

    B1 = (F_clr - F_all) / (F_clr + eps)
    B2 = (F_clr_drt - F_all_drt) / (F_clr_drt + eps)
    B3 = B1 / (B2 + eps)

    # compute metrics
    metrics_list = ['tot','dir']
    #metrics_list = ['dif','dir']

    
    # old style
    Chi = 0
    Chi_rgm = 0

    for var in metrics_list:
        theta_mod = 0
        theta_def = 0

        theta_mod_rgm = 0
        theta_def_rgm = 0

        for j in range(data_obs[var].shape[0]):
            if B1[j] > 0 and B1[j] <= 1 and B2[j] > 0 and B2[j] <= 1 and B3[j] > 0.007872 and B3[j] <= 1:
                theta_mod_rgm += abs(data_obs[var][j] - data_mod[var][j])
                theta_def_rgm += abs(data_obs[var][j] - data_def[var][j])

            theta_mod += abs(data_obs[var][j] - data_mod[var][j])
            theta_def += abs(data_obs[var][j] - data_def[var][j])


        #print(var, theta_mod / theta_def)a
        Chi_rgm += (theta_mod_rgm / theta_def_rgm)
        Chi += (theta_mod / theta_def)

    Chi_rgm /= len(var_list)
    Chi /= len(var_list)
    

    return Chi_rgm, Chi
        
if __name__ == '__main__':
    aa = chi_metrics(path_base+"wrfout_d02_2016-06-19_06_00_00")
    print(aa)
