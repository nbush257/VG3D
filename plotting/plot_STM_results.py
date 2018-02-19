import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()
sns.set_style('ticks')
# ===================== # 
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['font.sans-serif'] = 'Arial'
dpi_res = 600
fig_width = 6.9 # in
sns.set_style('ticks')
fig_height = 9 # in
ext = 'png'
p_save = r'C:\Users\guru\Box Sync\__VG3D\_deflection_trials\_NEO\results'
# ============================= #
# ============================= #
# 
# ========= No Hist =========== #
# fname = '/projects/p30144/_VG3D/deflections/_NEO/results/no_hist_correlations.csv'
fname = r'C:\Users\guru\Box Sync\__VG3D\_deflection_trials\_NEO\results\no_hist_correlations.csv'
is_stim = pd.read_csv(r'C:\Users\guru\Box Sync\__VG3D\_deflection_trials\_NEO\results\cell_id_stim_responsive.csv')
df = pd.read_csv(fname)
df = df.merge(is_stim,on='id')
df = df[df.stim_responsive]
cols = df.columns.tolist()
[cols.pop(cols.index(x)) for x in ['kernels','id','stim_responsive']]
cols.sort()
df2 =df.melt(id_vars=['id','kernels'],value_vars=cols,var_name='model_type',value_name='Pearson_Correlation')

# ======================
# Temporal accuracy of all models
wd = fig_width
ht = fig_width/3
sns.factorplot(x='model_type',y='Pearson_Correlation',data=df2,
               hue='kernels',
               kind='box',
               palette='Greens',
               legend=False,
               whis=1)
f = plt.gcf()
f.set_size_inches(wd,ht)
plt.legend(bbox_to_anchor=(1,1.1))
plt.xlabel('')
plt.ylabel('Pearson Correlation (R)')
plt.title('Temporal accuracy of models')
plt.tight_layout()
plt.savefig(os.path.join(p_save,'STM_all_models_kernels.{}'.format(ext)),dpi=dpi_res,bbox_inches='tight')
# ==============================
# Only at one smmothing param
wd = fig_width/2
ht = wd/1.2
kernel=16
cmap = [sns.palettes.color_palette('Paired',8)[ii] for ii in[1,3,5,7,0,2,4,6]]
cmap = [(.6,.6,.6)]+cmap
order=['full','justM','justF','justR','justD','noM','noF','noR','noD']
plt.figure(figsize=(wd,ht))
sns.boxplot(x='model_type',y='Pearson_Correlation',data=df2[df2.kernels==kernel],fliersize=0,whis=1,palette=cmap,order=order)
sns.swarmplot(x='model_type',y='Pearson_Correlation',data=df2[df2.kernels==kernel],color='k',alpha=0.6,size=2,order=order)
ax = plt.gca()
ax.set_ylim(-0.05,1)
plt.ylabel('Pearson Correlation (R)')
plt.xlabel('')
sns.despine(offset=10)
plt.xticks(range(9),['Full','Only Bending','Only Lateral','Only Rotation','Only Derivative',
                     'Drop Bending','Drop Lateral','Drop Rotation','Drop Derivative'],rotation=45,horizontalalignment='right')
plt.tight_layout()
plt.savefig(os.path.join(p_save,'all_models_{}_ms.{}'.format(kernel,ext)),dpi=dpi_res,bbox_inches='tight')
# ============================
# Derivative information improves temporal accuracy
wd = fig_width/2
ht = wd/.8
plt.figure(figsize=(wd,ht))
df_full = pd.pivot_table(df[['full','kernels','id']],columns='kernels',index='id',values='full')
df_noD = pd.pivot_table(df[['noD','kernels','id']],columns='kernels',index='id',values='noD')
sns.distplot(df_full.T.idxmax(),kde=False,bins=np.power(2,range(1,10)))
sns.distplot(df_noD.T.idxmax(),kde=False,bins=np.power(2,range(1,10)))
ax = plt.gca()
ax.legend(['Full','No Derivative'],bbox_to_anchor=(.75,.75))
ax.set_xscale("log")
ax.set_xticks(np.power(2,range(1,10))+np.power(2,range(9)))
ax.set_xticklabels(np.power(2,range(1,10)),rotation=60)
ax.minorticks_off()
sns.despine()
plt.xlabel('Best smoothing window (ms)')
plt.ylabel('Number of cells')
plt.title('Derivative information improves\ntemporal accuracy of models')
plt.tight_layout()
plt.savefig(os.path.join(p_save,'derivative_temporal_accuracy_hist.{}'.format(ext)),dpi=dpi_res,bbox_inches='tight')
#
wd = fig_width/2
ht = fig_width/2
df3 =df.melt(id_vars=['id','kernels'],value_vars=['full','noD'],var_name='model_type',value_name='Pearson_Correlation')
sns.factorplot(x='model_type',y='Pearson_Correlation',data=df3,
              hue='kernels',
               kind='box',
               palette='Greens',
               legend=False)
f = plt.gcf()
f.set_size_inches(wd,ht)
plt.xlabel('')
plt.ylabel('Pearson Correlation (R)')
plt.title('Temporal accuracy of models')
sns.despine(offset=10)
plt.tight_layout()
plt.savefig(os.path.join(p_save,'derivative_temporal_accuracy.{}'.format(ext)),dpi=dpi_res)
# =======================================
# Difference from full
wd = fig_width/3
ht = wd/.5
f = plt.figure(figsize=(wd,ht))

sub_df = df[df.kernels==kernel]
df_diff = sub_df[['noM','noF','noR','noD']].subtract(sub_df['full'],axis=0)
df_pct_diff = df_diff.div(sub_df['full'],axis=0)
sns.boxplot(data=df_pct_diff[['noM','noF','noR']],fliersize=0)
sns.swarmplot(data=df_pct_diff[['noM','noF','noR']],color='k',size=2.5,alpha=0.5)
sns.despine()
ax =plt.gca()
y_anot =ax.get_ylim()[1]
offset=.05
plt.annotate('n.s.',xy=(0,y_anot),horizontalalignment='center')
plt.annotate('n.s.',xy=(1,y_anot),horizontalalignment='center')
plt.annotate('*',xy=(2,y_anot),horizontalalignment='center')
plt.ylabel('$\\frac{R_{full}-R_{subset}}{R_{full}}$')
plt.xticks([0,1,2],['Drop Bending','Drop Lateral','Drop Rotation'],rotation=60)
plt.tight_layout()
plt.savefig(os.path.join(p_save,'percent_diff_full_drops.{}'.format(ext)),dpi=dpi_res)
# TODO: Is the percent difference correlated? For neurons that need M, can they do without R?
# =================================
# Single Types
wd = fig_width/2
ht=wd
f = plt.figure(figsize=(wd,ht))

plt.subplot(221)
plt.plot(sub_df['justM'],sub_df['justF'],'.',color='k',alpha=0.75)
plt.plot([0,1],[0,1],'r--')
sns.despine()
plt.axis('square')
plt.ylim(0,1)
plt.xlim(0,1)
plt.ylabel('Just Force')
plt.subplot(223)
plt.plot(sub_df['justM'],sub_df['justR'],'.',color='k',alpha=0.75)
plt.plot([0,1],[0,1],'r--')
sns.despine()
plt.axis('square')
plt.ylim(0,1)
plt.xlim(0,1)
plt.ylabel('Just Rotation')
plt.xlabel('Just Moment')
plt.subplot(224)
plt.plot(sub_df['justF'],sub_df['justR'],'.',color='k',alpha=0.75)
plt.plot([0,1],[0,1],'r--')
sns.despine()
plt.axis('square')
plt.ylim(0,1)
plt.xlim(0,1)
plt.xlabel('Just Force')
plt.tight_layout()
plt.savefig(os.path.join(p_save,'single_type_model_compare.{}'.format(ext)),dpi=dpi_res)
plt.close('all')
