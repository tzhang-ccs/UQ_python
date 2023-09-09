import os

path_para="/lcrc/group/e3sm/ac.tzhang/E3SMv3/20230223.NGD_v3atm.piControl.tune/"
path_archive="/home/ac.tzhang/fs0_large/E3SM_archive/workdir.8/"

os.system(f"mkdir -p {path_archive}/restart")

os.system(f"cp {path_para}/run/rpointer.*  {path_archive}/restart")
os.system(f"cp {path_para}/run/20230223.NGD_v3atm.piControl.tune.*.{{r,i}}*.0002-01-01*nc  {path_archive}/restart")
os.system(f"cp {path_para}/run/20230223.NGD_v3atm.piControl.tune.{{eam,elm}}.h0.0001-12*  {path_archive}/restart")
