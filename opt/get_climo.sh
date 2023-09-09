
case_id=F2010.ne30pg2_EC30to60E2r2_TuRBO_M3
input_path=/lcrc/group/e3sm/ac.tzhang/E3SMv2_1/$case_id/run/
output_path=/lcrc/group/e3sm/ac.tzhang/E3SMv2_1/$case_id/diags
map_file=map_files/map_ne30pg2_to_cmip6_180x360_aave.20200201.nc


mkdir -p $output_path/tmp
#ncclimo -s 1 -e 10 -c $case_id  -i $input_path  -o $output_path/tmp -r $map_file --no_amwg_links
ncclimo -v PRECC,PRECL,LWCF  -s 1 -e 2 -c $case_id  -i $input_path  -o $output_path/tmp  -O $output_path -r $map_file --no_amwg_links

rm -rf $output_path/tmp

