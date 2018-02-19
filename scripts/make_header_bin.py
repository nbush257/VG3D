import pandas as pd
import sys
binsize=sys.argv[1]
p_smooth = r'/projects/p30144/_VG3D/deflections/_NEO'
df_head = pd.DataFrame(columns=['id','full','noD','noM','noF','noR'])
csv_file = os.path.join(p_smooth,'{}_bin_model_correlations.csv'.format(binsize))
df_head.to_csv(csv_file,index=None)
