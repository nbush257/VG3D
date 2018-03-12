import scipy
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
df = pd.read_csv(os.path.join(p_save,r'STM_3D_PCA_correlations.csv'))
df_raw = pd.read_csv(os.path.join(p_save,r'STM_3D_correlations.csv'))

df = df.merge(is_stim,on='id')
df = df[df.stim_responsive]
cols = df.columns.tolist()
[cols.pop(cols.index(x)) for x in ['kernels','id','stim_responsive']]
cols.sort()
df2 =df.melt(id_vars=['id','kernels'],value_vars=cols,var_name='model_type',value_name='Pearson_Correlation')

df_raw = df_raw.merge(is_stim,on='id')
df_raw = df_raw[df_raw.stim_responsive]
cols_raw = df_raw.columns.tolist()
[cols_raw.pop(cols_raw.index(x)) for x in ['kernels','id','stim_responsive']]
cols_raw.sort()
df2_raw =df_raw.melt(id_vars=['id','kernels'],value_vars=cols_raw,var_name='model_type',value_name='Pearson_Correlation')
# =============================
# For edification:
sns.factorplot(x='model_type',y='Pearson_Correlation',data=df2,
               hue='kernels',
               kind='box',
               palette='Greens',
               legend=False,
               whis=1)
# =================================
wd = figsize[0]/2
ht = wd/0.66
kernel=16
order = ['full{}'.format(x) for x in range(6,0,-1)]
plt.figure(figsize=(wd,ht))
sns.boxplot(x='model_type',y='Pearson_Correlation',data=df2[df2.kernels==kernel],
            order=order,
            palette='Reds_r',
            whis=1,
            fliersize=0)
sns.stripplot(x='model_type',y='Pearson_Correlation',data=df2[df2.kernels==kernel],
              order=order,
              color='k',
              jitter=0.05,
              alpha=0.3)
sns.despine(offset=5)
plt.grid('on',axis='y')
plt.yticks([0,.5,1])
plt.ylabel('Pearson Correlation')
plt.xlabel('Number of components (original dim=8)')
plt.title('PCA models')
plt.tight_layout()
plotVG3D.savefig(p_save,'PCA_model_dropout.{}'.format(ext),dpi=dpi_res)
# plot full model
wd = ht

plt.figure(figsize=(wd,ht))
sns.boxplot(df2_raw[(df2_raw.model_type=='full') & (df2_raw.kernels==kernel)].Pearson_Correlation,
            orient='v',
            color=[0.2,0.2,0.2])
sns.stripplot(df2_raw[(df2_raw.model_type=='full') & (df2_raw.kernels==kernel)].Pearson_Correlation,
              color='k',
              alpha=0.8,
              jitter=0.05,
              orient='v')

# ==================================
# Diff from full model
wd = figsize[0]/2.5
ht = wd/0.5
f= plt.figure(figsize=(wd,ht))
df2 = df2.append(df2_raw[df2_raw.model_type=='full'])
df_pivot = pd.pivot_table(df2,index='id',columns='model_type',values='Pearson_Correlation')
pct_diff = df_pivot.subtract(df_pivot.full,axis=0).divide(df_pivot.full,axis=0)*100
sub_pct_diff = pct_diff[['full6','full5','full4','full3','full2','full1']]
sns.boxplot(data=sub_pct_diff,
            palette='Reds_r',
            whis=1,
            fliersize=0)

pvals = scipy.stats.ttest_1samp(sub_pct_diff,0)[1]
sig = pvals<(0.5/5)
ax = plt.gca()
for ii in range(6):
    if sig[ii]:
        txt='*'
    else:
        txt='n.s.'

    ax.annotate(txt,xy=(ii,25),horizontalalignment='center')
plt.ylim(-120,25)
sns.despine(offset=5)
plt.xlabel('')
plt.title('Percent difference\nfrom full model',y=1.1)
# plt.ylabel('$\\frac{R_{PCA}-R_{full}}{R_{full}}$')
plt.tight_layout()
plotVG3D.savefig(p_save,'pct_diff_PCA_drops_v_full.{}'.format(ext),dpi=dpi_res)

## =====================================================
# plot versus
dat = np.load(os.path.join(p_save,'cov_exp_var.npz'))
e =dat['exp_var']
var_sum = np.cumsum(e,axis=0)
df_pca = pd.read_csv(os.path.join(p_save,'STM_3D_PCA_correlations.csv'))

eigenvalues = pd.DataFrame()
eigenvalues['whisker'] = dat['id']
for ii in range(8):
    eigenvalues['cum_var_{}pcs'.format(ii)] = var_sum[ii, :]

df_pca['whisker'] = [id[:-2] for id in df_pca['id']]
df = df_pca.merge(eigenvalues,on='whisker')
df = df.merge(is_stim, on='id')
df = df[df.stim_responsive]

df_sub = df[df.kernels==16]
def reshape_pcs(df_sub):
    mdl_list = ['full{}'.format(ii) for ii in range(1,7)]
    eigenvalue_list = ['cum_var_{}pcs'.format(ii) for ii in range(0,6)]
    id_list = df_sub.id.unique()
    DF = pd.DataFrame()
    for id in id_list:
        df_temp = pd.DataFrame()
        pc = df_sub[df_sub.id == id][eigenvalue_list]
        accuracy = df_sub[df_sub.id == id][mdl_list]
        df_temp['Pearson_Correlation']=accuracy.values.ravel()
        df_temp['var_accounted_for']=pc.values.ravel()
        df_temp['id'] = id
        DF = DF.append(df_temp)
    return(DF)
df_long = reshape_pcs(df_sub)
sns.tsplot(x='var_accounted_for',y='Pearson_Correlation',data=df_long,condition='id')