import seaborn as sns
import plotVG3D
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
kernel=8
dpi_res,figsize,ext = plotVG3D.set_fig_style()
# =================================
p_save = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')
df_FR = pd.read_csv(os.path.join(p_save,'direction_arclength_FR_group_data.csv'))
df = pd.read_csv(os.path.join(p_save,'STM_3D_correlations.csv'))
df_reg = pd.read_csv(os.path.join(p_save,'regularity_by_contact.csv'))
is_stim = pd.read_csv(os.path.join(p_save,'cell_id_stim_responsive.csv'))
df = df.merge(is_stim,on='id')
df = df[df.stim_responsive]
#====================================
df_reg = df_reg.groupby('id').mean()
df_FR = df_FR.groupby('id').mean()

df_sub = df[df.kernels==kernel][['id','full']]
df_sub = df_sub.merge(df_FR,left_on='id',right_index=True)
df_sub = df_sub.merge(df_reg[['LV','CV']],left_on='id',right_index=True)
# sns.pairplot(df_sub[['full','Firing_Rate','LV']].dropna(),
#              diag_kind='kde',
#              diag_kws={'shade':True,
#                        'color':'k'},
#              plot_kws={'color':'k',
#                        'alpha':0.6})
#
# plt.suptitle('Kernel Smoothing = {} ms'.format(kernel))
# plt.savefig(os.path.join(p_save,'regularity_performance_pairplot.{}'.format(ext)),dpi=dpi_res)
# ==========================
plt.figure(figsize=(figsize[0]/2.5,figsize[0]/2.5))
cmap = sns.cubehelix_palette(light=0.9,as_cmap=True)
plt.scatter(df_sub.Firing_Rate,df_sub.full,c=df_sub.LV,edgecolor='k',cmap=cmap)
cbar = plt.colorbar(pad=0.2)
cbar.set_clim(vmin=-0.1)
cbar.set_ticks(np.arange(0,cbar.get_clim()[-1],0.5))
cbar.set_label('Spiking Irregularity')
sns.despine()
plt.ylabel('Model Accuracy (R)')
plt.xlabel('Firing Rate')
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(os.path.join(p_save,'regularity_performance_color_scatter.{}'.format(ext)),dpi=dpi_res)
plt.close('all')


