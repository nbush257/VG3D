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
df_2d = pd.read_csv(os.path.join(p_save,r''))
df_3d = pd.read_csv(os.path.join(p_save,r''))

df_2d = df_2d.merge(is_stim,on='id')
df_3d = df_3d.merge(is_stim,on='id')

df_2d = df_2d[df_2d.stim_responsive]
df_3d = df_3d[df_3d.stim_responsive]

cols = df.columns.tolist()
[cols.pop(cols.index(x)) for x in ['kernels','id','stim_responsive']]
cols.sort()

