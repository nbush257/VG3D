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
p_results = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')

# Load in all the pillow results
df_pillow_55ms = pd.read_csv(os.path.join(p_results,'pillow_MID_performance_55ms.csv'),index_col=0)
df_pillow_drops = pd.read_csv(os.path.join(p_results,'pillow_best_deriv_drop_correlations.csv'))
df_pillow_2D = pd.read_csv(os.path.join(p_results,'pillow_best_deriv_2D_correlations.csv'))
# Load in all the DTM results
df_STM = pd.read_csv(os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results\STM_3D_correlations.csv'))
df_STM = df_STM[['id','full','kernels']]
# load in the stim response and remove non-stim responsive
is_stim = pd.read_csv(os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results\cell_id_stim_responsive.csv'))
df_pillow_55ms = df_pillow_55ms.merge(is_stim,on='id')
df_pillow_55ms = df_pillow_55ms[df_pillow_55ms.stim_responsive]

df_pillow_drops = df_pillow_drops.merge(is_stim,on='id')
df_pillow_drops = df_pillow_drops[df_pillow_drops.stim_responsive]

df_pillow_2D = df_pillow_2D.merge(is_stim,on='id')
df_pillow_2D = df_pillow_2D[df_pillow_2D.stim_responsive]

df_STM = df_STM.merge(is_stim,on='id')
df_STM = df_STM[df_STM.stim_responsive]

#
df_merged = pd.merge(df_pillow_55ms,df_STM,how='left',left_on=['id','kernel_sizes'],right_on=['id','kernels'])
df_merged.drop(['stim_responsive_x','kernel_sizes'],axis = 1,inplace=True)
df_merged.rename(columns={'R':'Pillow_55ms','full':'STM','stim_responsive_y':'stim_responsive'},inplace=True)


df_merged.dropna(inplace=True)
df_merged.kernels = df_merged.kernels.astype('int')

df_melt = df_merged.melt(id_vars=['id','kernels'],value_vars=['Pillow_55ms','STM'],var_name=['model'],value_name='Pearson Correlation')
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
f = plt.gcf()
f.set_size_inches(wd,ht)
sns.despine()
plt.ylim(0,1.1)
plt.yticks([0,.5,1])
plt.xticks(rotation=45)
plt.title('Pillow model does better longer time\nPillow has slow derivative only\nSTM has 3 derivatives')
plt.tight_layout()
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
# ======================
# plot pct_diff (how much better the pillow is)
wd = figsize[0]/1.5
ht = figsize[0]/1.5
df_merged= df_merged.merge(df_pillow_drops[['full','id','kernels','stim_responsive']],
                            on=['id','kernels','stim_responsive'])
df_merged.rename(columns={'full':'Pillow Best'},inplace=True)
df_merged['Percent Difference']= df_merged['Pillow Best'].subtract(df_merged['STM'])
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
plt.title('Pillow Models are doing better\nPillow only has best derivative\nSTM has 3 derivatives')
plt.tight_layout()
# ==================================================
# =======  Compare best deriv against 55ms =========
# ==================================================
kernel=32
df_sub_drop = df_pillow_drops[['full','id']][df_pillow_drops.kernels==kernel]
df_sub_55 = df_pillow_55ms[['R','id']][df_pillow_55ms.kernel_sizes==kernel]
df_sub_55.rename(columns={'R':'55'},inplace=True)
df_sub_drop.rename(columns={'full':'best'},inplace=True)
df_temp = df_sub_drop.merge(df_sub_55,on='id')

f = plt.figure()
sns.pointplot(data=df_temp,color='k')
sns.stripplot(data=df_temp,color='k',alpha=0.2,jitter=True)
plt.ylim(0,1)
sns.despine()
plt.tight_layout()

pct_diff = (df_temp['best']-df_temp['55'])/df_temp['best']

plt.figure()
thresh=0.2
idx = df_temp['best']>thresh
sns.boxplot(y=pct_diff[idx],color='w',whis=1,fliersize=0)
sns.stripplot(y=pct_diff[idx],color='k',alpha=0.6,jitter=True)
plt.xticks([0],['(Best-55)/Best'])
plt.ylabel('Percent Difference of Best 55ms smothing model versus best\nNegative means 55ms did worse')
sns.despine()
plt.tight_layout()


stats_diff = scipy.stats.ttest_rel(df_temp['best'][idx],df_temp['55'][idx])[1]

# ==================================================
# ===========  Plot Pillow Drops ===================
# ==================================================
smooth_param=32
order = ['full','noR','noM','noF','noD','justR','justM','justF','justD']
df_drops_melt = df_pillow_drops.drop('stim_responsive',axis=1).melt(
    id_vars=['id','kernels'],
    var_name='Model',
    value_name='Pearson Correlation')
# all factors
sns.factorplot(data=df_drops_melt,
               x='Model',
               y='Pearson Correlation',
               hue='kernels',
               kind='box',
               palette='Greens',
               order=order)
# just one smoothing param
plt.figure()
df_sub = df_drops_melt[df_drops_melt.kernels==smooth_param]
sns.boxplot(x='Model',
            y='Pearson Correlation',
            data=df_sub,
            order=order,
            whis=1,
            fliersize=0)
sns.stripplot(x='Model',
              y='Pearson Correlation',
              data=df_sub,
              order=order,
              jitter=True,
              color='k',
              alpha=0.3)
sns.despine()


# =================================================
# Percent Difference Pillow
sub_df = df_pillow_drops[df_pillow_drops.kernels==smooth_param]
df_diff = sub_df[['noM','noF','noR','noD']].subtract(sub_df['full'],axis=0)
df_pct_diff = df_diff.div(sub_df['full'],axis=0)
thresh=0.2
idx = sub_df['full']>thresh
sns.boxplot(data=df_pct_diff[idx]*100.,fliersize=0)
sns.stripplot(data=df_pct_diff[idx]*100.,
              color='k',
              alpha=0.5,
              jitter=True)
sns.despine()
plt.grid('on',axis='y')
ax =plt.gca()
offset=.05
plt.ylabel('$\\frac{R_{full}-R_{subset}}{R_{full}}$')
plt.xticks([0,1,2,3],['Drop Bending','Drop Force','Drop Rotation','Drop Derivative'],rotation=60,horizontalalignment='right')
plt.tight_layout()

# =================================================
# Derivative information improves temporal accuracy
wd = figsize[0]/2
ht = wd/.8
plt.figure(figsize=(wd,ht))
df = df_pillow_drops
df_full = pd.pivot_table(df[['full','kernels','id']],columns='kernels',index='id',values='full')
df_noD = pd.pivot_table(df[['noD','kernels','id']],columns='kernels',index='id',values='noD')
idx = np.logical_and(
    np.all(np.isfinite(df_full),axis=1),
    np.all(np.isfinite(df_noD),axis=1),
)
sns.distplot(df_full.T.idxmax()[idx],kde=False,bins=np.power(2,range(1,10)))
sns.distplot(df_noD.T.idxmax()[idx],kde=False,bins=np.power(2,range(1,10)))
ax = plt.gca()
ax.set_xscale("log")
ax.set_xticks(np.power(2,range(1,10))+np.power(2,range(9)))
ax.set_xticklabels(np.power(2,range(1,10)),rotation=60)
ax.minorticks_off()
sns.despine()
plt.xlabel('Best smoothing window (ms)')
plt.ylabel('Number of cells')
plt.title('Derivative information improves\ntemporal accuracy of models')
plt.tight_layout()
# Are these different? Yes, for both these tests. Is there a better test?
# scipy.stats.ttest_rel(df_full.T.idxmax()[idx],df_noD.T.idxmax()[idx])
# scipy.stats.ttest_rel(np.log2(df_full.T.idxmax()[idx]),np.log2(df_noD.T.idxmax()[idx]))
wd = figsize[0]/2
ht = figsize[0]/2
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
# =======================================
# ======== 2D vs 3D =====================
# =======================================
kernel = 32
DF = pd.merge(df_pillow_drops[['full','id','kernels','stim_responsive']],
              df_pillow_2D,
              on=['id','kernels','stim_responsive'])
DF_kernel = DF[DF.kernels==kernel]
DF_kernel.rename(columns={'full':'3D'},inplace=True)
DF_kernel['pct_improve'] = (DF_kernel['3D']-DF_kernel['2D'])/DF_kernel['2D']*100

wd = figsize[0]/3
ht = figsize[0]/3
f = plt.figure(figsize=(wd,ht))
sns.barplot(data=DF_kernel[['2D','3D']],
            estimator=np.median,
            palette='Set2')
sns.stripplot(data=DF_kernel[['2D','3D']],
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
wd = figsize[0]/4
ht = wd/0.5
f = plt.figure(figsize=(wd,ht))
# 2 points are huge, well above 400% improvement
thresh = 400
idx = DF_kernel['pct_improve']<thresh
plt.bar(0,np.median(DF_kernel['pct_improve']),width=0.4,color='k',alpha=0.3,)
sns.stripplot(DF_kernel['pct_improve'][idx],orient='v',jitter=0.05,color='k',alpha=0.6,edgecolor='w')
plt.title('% improvement')
# plt.ylabel('$\\frac{R_{3D}-R_{2D}}{R_{3D}}$')
plt.ylabel('')
plt.xticks([])
sns.despine(offset=5)
plt.tight_layout()
# Stats:
pval_wilcoxon = scipy.stats.wilcoxon(DF_kernel['2D'],DF_kernel['3D'])[1]
pval_ttest = scipy.stats.ttest_rel(DF_kernel['2D'],DF_kernel['3D'])[1]
# A distribution:
# sns.distplot(DF_kernel['pct_improve'][idx],kde=False)

# =======================================
# ====== Single Types ==================
# =======================================
kernel =32
sub_df = df_pillow_drops[df_pillow_drops.kernels==kernel]
wd = figsize[0]/2
ht=wd
f = plt.figure(figsize=(wd,ht))

plt.subplot(221)
plt.plot(sub_df['justM'],sub_df['justF'],'.',color='k',alpha=0.75)
plt.plot([0,1],[0,1],'r--')
sns.despine()
plt.axis('square')
plt.ylim(0,1)
plt.xlim(0,1)
plt.xticks([0,0.5,1])
plt.yticks([0,0.5,1])
plt.ylabel('Just Force')
plt.subplot(223)
plt.plot(sub_df['justM'],sub_df['justR'],'.',color='k',alpha=0.75)
plt.plot([0,1],[0,1],'r--')
sns.despine()
plt.axis('square')
plt.ylim(0,1)
plt.xlim(0,1)
plt.xticks([0,0.5,1])
plt.yticks([0,0.5,1])
plt.ylabel('Just Rotation')
plt.xlabel('Just Moment')
plt.subplot(224)
plt.plot(sub_df['justF'],sub_df['justR'],'.',color='k',alpha=0.75)
plt.plot([0,1],[0,1],'r--')
sns.despine()
plt.axis('square')
plt.ylim(0,1)
plt.xlim(0,1)
plt.xticks([0,0.5,1])
plt.yticks([0,0.5,1])
plt.xlabel('Just Force')
plt.tight_layout()

# =======================================
# ====== Arclength Comparisons ==========
# =======================================
smooth_param=16
df_arclength = pd.read_csv(os.path.join(p_results,'pillow_best_deriv_arclengths_drop_correlations_melted.csv'))
sub_df = df_arclength[(df_arclength['kernels']==smooth_param) &
                      (df_arclength['Inputs'].isin(['Full','Drop_moment','Drop_rotation','Drop_force'])) &
                      df_arclength['Arclength'].isin(['all','distal','proximal'])]

sns.factorplot(x='Arclength',y='Correlation',data=sub_df,hue='Inputs',
               hue_order=['Full','Drop_moment','Drop_force','Drop_rotation'],
               kind='violin',
               palette='colorblind',
               size=5,
               aspect=1.5,
               legend=False)

plt.legend(bbox_to_anchor=(1,0.5))
plt.tight_layout()

sub_df = df_arclength[(df_arclength['kernels']==smooth_param) &
                      (df_arclength['Inputs'].isin(['Full','Just_moment','Just_rotation','Just_force'])) &
                      df_arclength['Arclength'].isin(['all','distal','proximal'])]
sns.factorplot(x='Arclength',y='Correlation',data=sub_df,hue='Inputs',
               hue_order=['Full','Just_moment','Just_force','Just_rotation'],
               kind='violin',
               palette='colorblind',
               size=5,
               aspect=1.5,
               legend=False)

plt.legend(bbox_to_anchor=(1,0.5))
plt.tight_layout()
