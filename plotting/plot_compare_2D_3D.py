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
# ===========================
# PREP DATA

is_stim = pd.read_csv(os.path.join(p_save,r'cell_id_stim_responsive.csv'))
df_2d = pd.read_csv(os.path.join(p_save,r'STM_2D_correlations.csv'))
df_3d = pd.read_csv(os.path.join(p_save,r'STM_3D_correlations.csv'))

df_2d = df_2d.merge(is_stim,on='id')
df_3d = df_3d.merge(is_stim,on='id')

df_2d = df_2d[df_2d.stim_responsive]
df_3d = df_3d[df_3d.stim_responsive]
id_intersect = np.intersect1d(df_2d.id.unique(),df_3d.id.unique())

cols = df_2d.columns.tolist()
[cols.pop(cols.index(x)) for x in ['kernels','id','stim_responsive']]
cols.sort()


df_2d['model']='2D'
df_3d['model']='3D'
DF = df_2d.append(df_3d)
DF = DF[DF.id.isin(id_intersect)]
DF =DF.melt(id_vars=['id','kernels','model'],value_vars=cols,var_name='model_type',value_name='Pearson_Correlation')
# ================================================================
## Plot comparison of 2D 3D for all drops at one smoothing param
wd = figsize[0]/1.5
ht = wd/1.5
kernel=16
sns.factorplot(data=DF[DF.kernels==kernel],
               x='model_type',
               y='Pearson_Correlation',
               hue='model',
               kind='box',
               palette='Set2')
f = plt.gcf()
ax = plt.gca()
ax.set_ylim([0,1])
f.set_size_inches((wd,ht))
sns.despine(offset=5,trim=True)

plt.tight_layout()

plt.figure(figsize=(wd,ht))
sns.violinplot(data=DF[DF.kernels==kernel],
               x='model_type',
               y='Pearson_Correlation',
               hue='model',
               split=True,
               palette='Set2',
               bw=0.3,
               inner='quartile'
               )
plt.ylim(0,1)
sns.despine(offset=5,trim=True)
plt.tight_layout()
# ==============================================================
compare = pd.pivot_table(DF_full_16,index='id',columns='model',values='Pearson_Correlation')
compare['pct_improve'] = (compare['3D']-compare['2D'])/compare['2D']*100

wd = figsize[0]/3
ht = figsize[0]/3
f = plt.figure(figsize=(wd,ht))
DF_full_16 = DF[(DF.kernels==kernel) & (DF.model_type=='full')]
sns.barplot(x='model',y='Pearson_Correlation',data=DF_full_16,
            estimator=np.median,
            palette='Set2')
sns.stripplot(x='model',y='Pearson_Correlation',data=DF_full_16,
              color='k',
              alpha=0.3,
              jitter=True)
# for cell in compare.index:
#     plt.plot([0,1],[compare.loc[cell,'2D'],compare.loc[cell,'3D']],'k-',alpha=0.2)

ax = plt.gca()
ax.set_ylim(-0.05,1)
plt.yticks([0,.5,1])
plt.ylabel('Pearson Correlation (R)')
plt.xlabel('')
plt.grid('on',axis='y')
sns.despine(trim=True,offset=5)
plt.tight_layout()
# ===================================================
# plot percent difference full
# sns.distplot(compare['pct_improve'],kde=False,bins=20)
wd = figsize[0]/3
ht = wd/0.75
f = plt.figure(figsize=(wd,ht))
# sns.barplot(compare['pct_improve'],orient='v')
sns.stripplot(compare['pct_improve'],orient='v',jitter=0.05,color='k',alpha=0.6,edgecolor='w')
plt.title('Percent improvement')
# plt.ylabel('$\\frac{R_{3D}-R_{2D}}{R_{3D}}$')
plt.ylabel('')
plt.xticks([])
sns.despine(offset=5)
plt.tight_layout()