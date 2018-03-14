import varTuning
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotVG3D
# ===================
dpi_res,figsize,ext=plotVG3D.set_fig_style()
p_save = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')
is_stim = pd.read_csv(os.path.join(p_save,r'cell_id_stim_responsive.csv'))
df = pd.read_csv(os.path.join(p_save,'regularity_by_contact.csv'))
df = df.merge(is_stim,on='id')
df = df[df.stim_responsive]
cell_list = ['201708D1c0']
# =======================================
if len(cell_list)==0:
    cell_list = df.id.unique()
# Plot regularity by direction for a cell
wd = figsize[0]/3
ht = wd
for cell in cell_list:
    sub_df =df[df.id==cell]
    df_dir = sub_df.groupby('dir_idx')
    theta = df_dir.med_dir.mean()
    # =========================
    plt.figure(figsize=(wd,ht))
    med_CV = df_dir.CV.quantile(0.5)
    lower_CV = df_dir.CV.quantile(0.25)
    upper_CV = df_dir.CV.quantile(0.75)
    theta[8]=theta[0]
    med_CV[8] = med_CV[0]
    lower_CV[8] = lower_CV[0]
    upper_CV[8] = upper_CV[0]
    plt.polar(theta,med_CV,alpha=0.8,color='k')
    plt.fill_between(theta,lower_CV,upper_CV,alpha=0.5,color='k')
    plt.title('CV by direction\n{}'.format(cell))
    plt.tight_layout()
    plotVG3D.savefig(p_save,'{}_CV_by_dir.{}'.format(cell,ext))
    # ==================
    plt.figure(figsize=(wd,ht))
    med_LV = df_dir.LV.quantile(0.5)
    lower_LV = df_dir.LV.quantile(0.25)
    upper_LV = df_dir.LV.quantile(0.75)
    med_LV[8] = med_LV[0]
    lower_LV[8] = lower_LV[0]
    upper_LV[8] = upper_LV[0]
    plt.polar(theta,med_LV,alpha=0.8,color='k')
    plt.fill_between(theta,lower_LV,upper_LV,alpha=0.5,color='k')
    plt.title('LV by direction\n{}'.format(cell))
    plt.tight_layout()
    plotVG3D.savefig(p_save,'{}_LV_by_dir.{}'.format(cell,ext))
# ====================================================
# Get DSI of CV regularity for all cells
wd = figsize[0]/2.5
ht = wd
df_dsi = pd.DataFrame()
DSI = []
ID = []
med_CV_all = []
for cell in df.id.unique():
    sub_df = df[df.id==cell]
    df_dir = sub_df.groupby('dir_idx')
    theta = df_dir.med_dir.mean()
    med_CV = df_dir.CV.quantile(0.5)
    DSI.append(varTuning.get_PD_from_hist(theta,med_CV)[1])
    ID.append(cell)
    med_CV_all.append(np.nanmedian((med_CV)))
df_dsi['id'] = ID
df_dsi['DSI'] = DSI
df_dsi['med_CV'] = med_CV_all

f = plt.figure(figsize=(wd,ht))
sns.distplot(df_dsi.DSI[np.isfinite(df_dsi.DSI)],20,kde=False,color='k')
sns.despine()
plt.title('Most cells regularity\nis not direction dependant')
plt.ylabel('Number of Cells')
plt.xlabel('Direction Selectivity of ISI CV')
plt.tight_layout()
plotVG3D.savefig(p_save,'DSI(CV)_all_cells.{}'.format(ext))
# plot CV vs DSI(cv)
f = plt.figure(figsize=(wd,ht))
sns.jointplot(x='med_CV',y='DSI',data=df_dsi,color='k')
plt.tight_layout()
plotVG3D.savefig(p_save,'DSI(CV)_vs_CV.{}'.format(ext))
# There is only a weak relatioship between the CV and the directionality of the CV
# =============================================================
# Get DSI of LV,CV, and pct_finite for all cells
df_dsi = pd.DataFrame()
DSI_LV = []
DSI_CV = []
DSI_pct_finite = []
ID = []
med_LV_all = []
med_CV_all = []
pct_finite_all = []
for cell in df.id.unique():
    sub_df = df[df.id==cell]
    sub_df['did_spike'] = np.isfinite(sub_df['mean_ISI'])
    df_dir = sub_df.groupby('dir_idx')
    theta = df_dir.med_dir.mean()

    med_LV = df_dir.LV.quantile(0.5)
    med_CV = df_dir.CV.quantile(0.5)
    pct_finite = df_dir.did_spike.mean()

    DSI_pct_finite.append(varTuning.get_PD_from_hist(theta,pct_finite)[1])
    DSI_LV.append(varTuning.get_PD_from_hist(theta,med_LV)[1])
    DSI_CV.append(varTuning.get_PD_from_hist(theta,med_CV)[1])

    med_CV_all.append(np.nanmedian((med_CV)))
    med_LV_all.append(np.nanmedian((med_LV)))
    pct_finite_all.append(np.nanmean(pct_finite))

    ID.append(cell)

df_dsi['id'] = ID
df_dsi['DSI_LV'] = DSI_LV
df_dsi['med_LV'] = med_LV_all
df_dsi['DSI_CV'] = DSI_CV
df_dsi['med_CV'] = med_CV_all
df_dsi['DSI_pct_finite'] = DSI_pct_finite
df_dsi['pct_finite'] = pct_finite_all
# ===========================
# Plot DSI by LV
wd = figsize[0]/2.5
ht = wd
f = plt.figure(figsize=(wd,ht))
sns.distplot(df_dsi.DSI_LV[np.isfinite(df_dsi.DSI_LV)],20,kde=False,color='k')
sns.despine()
plt.title('Maybe 3 clusters of LV')
plt.ylabel('Number of Cells')
plt.xlabel('Direction Selectivity of ISI LV')
plt.tight_layout()
plotVG3D.savefig(p_save,'DSI(LV)_all_cells.{}'.format(ext))
# plot LV vs DSI(LV)
sns.jointplot(x='med_LV',y='DSI_LV',data=df_dsi,color='k')
plt.tight_layout()
plotVG3D.savefig(p_save,'DSI(LV)_vs_LV.{}'.format(ext))

# There is no relationship between the LV and the directionality of the LV
# =============================================================
# Plot distribution of regularity for all cells (as horizontal box plots)
ht = figsize[1]
wd = ht/3
f = plt.figure(figsize=(wd,ht))
df_by_id =pd.pivot_table(df,columns='dir_idx',values=['CV','LV'],index='id')
cmap = sns.cubehelix_palette(start=3,light=0.85,dark=0.15)
sns.boxplot(data=df_by_id['LV'].T,
            orient='h',
            order=df_by_id['LV'].median(axis=1).sort_values().index,
            whis=1,
            fliersize=3) #probably doesn't need color
sns.despine(trim=True)
plt.grid('on',axis='x')
plt.xlabel('Regularity (LV)')
plt.ylabel('Cell (ordered by LV)')
ax = plt.gca()
ax.set_yticklabels('')
plt.tight_layout()
plotVG3D.savefig(p_save,'all_cells_LV_dist.{}'.format(ext),dpi=dpi_res)
# ==================================================
# Plot polar of pct of deflections with >2 spikes
wd = figsize[0]/2
ht=wd
for cell in cell_list:
    sub_df = df[df.id==cell]
    sub_df['mean_ISI'] = np.isfinite(sub_df['mean_ISI'])
    pct_finite = sub_df.groupby('dir_idx')['mean_ISI'].mean()
    theta = sub_df.groupby('dir_idx')['med_dir'].mean()
    pct_finite[8]=pct_finite[0]
    theta[8]=theta[0]
    f = plt.figure(figsize=(wd,ht))
    plt.polar(theta,pct_finite,alpha=0.8,color='k',linestyle=':')
    plt.fill_between(theta,np.zeros(len(theta)),pct_finite,alpha=0.4,color='k')
    plt.title('{}\nPercent of deflections\nwith more than 2 spikes\nDSI={:0.2f}'.format(cell,df_dsi[df_dsi.id==cell].DSI_pct_finite.values[0]))
    ax = plt.gca()
    ax.set_rlim(0,1)
    ax.set_rticks([0.5,1])
    plt.tight_layout()
    plotVG3D.savefig(p_save,'{}_spikethresh_by_direction.{}'.format(cell,ext),dpi=dpi_res)

# =================================
# Plot DSI of percentage of contacts with less than 3 spikes by direction
# working on removing this part
f = plt.figure(figsize=(wd,ht))
sns.distplot(df_dsi.DSI_pct_finite[np.isfinite(df_dsi.DSI_pct_finite)],20,kde=False,color='k')
sns.despine()
plt.title('How directional is the probability to spike?')
plt.ylabel('Number of Cells')
plt.xlabel('Direction selectivity of probability of >1 spike')
plt.tight_layout()
# plot pct finite vs DSI(pct finite)
sns.jointplot(x='pct_finite',y='DSI_pct_finite',data=df_dsi,color='k')
plt.tight_layout()
# =================================
# is there a relationship between regularity directionality and threshold directionality?
# Yes
wd = figsize[0]/1.5
ht = figsize[0]/1.5
f = (sns.jointplot(df_dsi['DSI_LV'],df_dsi['DSI_pct_finite'],
              kind='scatter',
              color='k',
              size=2,
              edgecolor='w',
              alpha=0.6,
              marginal_kws=dict(bins=20, kde=False))).set_axis_labels('Directionality of regularity(LV)','Directionality of threshold')
f = plt.gcf()
f.set_size_inches(wd,ht)
sns.despine()
plt.tight_layout()
plotVG3D.savefig(p_save,'DSI(probability_of_spike)_all_cells.{}'.format(ext),dpi=dpi_res)
plt.close('all')