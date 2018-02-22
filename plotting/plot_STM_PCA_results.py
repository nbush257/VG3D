import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotVG3D
# ===================== #
dpi_res,figsize,ext=plotVG3D.set_fig_style()
p_save = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')
fname = os.path.join(p_save,r'no_hist_correlations.csv')
is_stim = pd.read_csv(os.path.join(p_save,r'cell_id_stim_responsive.csv'))
df = pd.read_csv(os.path.join(p_save,r''))

df = df.merge(is_stim,on='id')
df = df[df.stim_responsive]
cols = df.columns.tolist()
[cols.pop(cols.index(x)) for x in ['kernels','id','stim_responsive']]
cols.sort()
df2 =df.melt(id_vars=['id','kernels'],value_vars=cols,var_name='model_type',value_name='Pearson_Correlation')
