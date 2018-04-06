import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import plotVG3D
import os

figsize = plotVG3D.set_fig_style()[1]
# ======================================
# Munge data
# ======================================
p_load = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\pillowX')
p_results = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')

df_pillow = pd.read_csv(os.path.join(p_load,'pillow_MID_performance.csv'),index_col=0)
df_STM = pd.read_csv(os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results\STM_3D_correlations.csv'))
df_STM = df_STM[['id','full','kernels']]
is_stim = pd.read_csv(os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results\cell_id_stim_responsive.csv'))
df_pillow = df_pillow.merge(is_stim,on='id')
df_pillow = df_pillow[df_pillow.stim_responsive]

df_STM = df_STM.merge(is_stim,on='id')
df_STM = df_STM[df_STM.stim_responsive]

df_merged = pd.merge(df_pillow,df_STM,how='left',left_on=['id','kernel_sizes'],right_on=['id','kernels'])
df_merged.drop(['stim_responsive_x','kernel_sizes'],axis = 1,inplace=True)
df_merged.rename(columns={'R':'Pillow','full':'STM','stim_responsive_y':'stim_responsive'},inplace=True)
df_merged.dropna(inplace=True)
df_merged.kernels = df_merged.kernels.astype('int')

df_melt = df_merged.melt(id_vars=['id','kernels'],value_vars=['Pillow','STM'],var_name=['model'],value_name='Pearson Correlation')
df_melt.dropna(inplace=True)
df_melt.kernels = df_melt.kernels.astype('int')

# ==========================================
# Plot comparisons
# ==========================================
wd = figsize[0]/2.5
ht = figsize[0]/2
sns.factorplot(
    data=df_melt,
    x='kernels',
    y='Pearson Correlation',
    hue='model',
    palette='Set2',
    markers='.',
    legend_out=False,
    linewidth=2
)
# sns.factorplot(
#     data=df_melt,
#     x='kernels',
#     y='Pearson Correlation',
#     hue='model',
#     palette='Set2',
#     legend=False,
#     kind='strip',
#     alpha=0.4,
#     jitter=True
# )
f = plt.gcf()
f.set_size_inches(wd,ht)
sns.despine()
plt.ylim(0,1.1)
plt.yticks([0,.5,1])
plt.xticks(rotation=45)
plt.title('Pillow model does better longer time\nPillow has slow derivative only\nSTM has 3 derivatives')
plt.tight_layout()
plt.savefig(os.path.join(p_results,'compare_pillow_stm.pdf'))
# ==========================
f = plt.figure(figsize=(wd,ht))
sns.violinplot(
    data=df_melt,
    x='kernels',
    y='Pearson Correlation',
    hue='model',
    split=True,
    inner='quartile',
    palette='Set2',
    legend_out=False
)
sns.despine()
plt.grid('on',axis='y')
plt.yticks([0,.5,1])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(p_results,'compare_pillow_stm_violin.pdf'))
# ======================
# plot pct_diff (how much better the pillow is)
wd = figsize[0]/1.5
ht = figsize[0]/1.5
df_merged['Percent Difference']= df_merged['Pillow'].subtract(df_merged['STM'])
df_merged['Percent Difference'] = df_merged['Percent Difference'].divide(df_merged['STM'])*100
f = plt.figure(figsize=(wd,ht))
sns.stripplot(
    data=df_merged,
    x='kernels',
    y='Percent Difference',
    palette='Greens',
    edgecolor='k',
    linewidth=1,
    jitter=True,
    alpha=0.5
)
sns.boxplot(
    data=df_merged,
    x='kernels',
    y='Percent Difference',
    whis=1,
    fliersize=0,
    palette='Greens',
    linewidth=0.5,
    )
sns.despine()
plt.axhline(c='k',ls='--')
plt.grid('on',axis='y')
plt.title('Pillow Models are doing better\nPillow only has slow derivative\nSTM has 3 derivatives')
plt.tight_layout()
plt.savefig(os.path.join(p_results,'Pct_diff_pillow_stm.pdf'))