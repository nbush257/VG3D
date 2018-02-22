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
df = pd.read_csv(os.path.join(p_save,'regularity_by_contact.csv'))
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
# plot CV vs DSI(cv)
f = plt.figure(figsize=(wd,ht))
sns.jointplot(x='med_CV',y='DSI',data=df_dsi,color='k')
plt.tight_layout()
# There is only a weak relatioship between the CV and the directionality of the CV
# =============================================================
# Get DSI of LV regularity for all cells
wd = figsize[0]/2.5
ht = wd
df_dsi = pd.DataFrame()
DSI = []
ID = []
med_LV_all = []
for cell in df.id.unique():
    sub_df = df[df.id==cell]
    df_dir = sub_df.groupby('dir_idx')
    theta = df_dir.med_dir.mean()
    med_LV = df_dir.LV.quantile(0.5)
    DSI.append(varTuning.get_PD_from_hist(theta,med_LV)[1])
    ID.append(cell)
    med_LV_all.append(np.nanmedian((med_LV)))
df_dsi['id'] = ID
df_dsi['DSI_med_LV'] = DSI
df_dsi['med_LV'] = med_LV_all

f = plt.figure(figsize=(wd,ht))
sns.distplot(df_dsi.DSI_med_LV[np.isfinite(df_dsi.DSI_med_LV)],20,kde=False,color='k')
sns.despine()
plt.title('Maybe 3 clusters of LV')
plt.ylabel('Number of Cells')
plt.xlabel('Direction Selectivity of ISI LV')
plt.tight_layout()
# plot LV vs DSI(cv)
f = plt.figure(figsize=(wd,ht))
sns.jointplot(x='med_LV',y='DSI_med_LV',data=df_dsi,color='k')
plt.tight_layout()
# There is no relationship between the LV and the directionality of the LV
# =============================================================

# Plot distribution of regularity for all cells (as horizontal box plots)
wd = figsize[0]/2
ht = wd/0.33
f = plt.figure(figsize=(wd,ht))
df_by_id =pd.pivot_table(df,columns='dir_idx',values=['CV','LV'],index='id')
sns.boxplot(data=df_by_id['LV'].T,orient='h',order=df_by_id['LV'].median(axis=1).sort_values().index,whis=1,fliersize=3) #probably doesn't need color
sns.despine(trim=True)
plt.grid('on',axis='x')
plt.xlabel('Regularity (LV)')
plt.ylabel('Cell (ordered by LV)')
ax = plt.gca()
ax.set_yticklabels('')
plt.tight_layout()
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
    plt.title('Percent of deflections\nwith more than 2 spikes'.format(cell))
    ax = plt.gca()
    ax.set_rlim(0,1)
    ax.set_rticks([0.5,1])
    plt.tight_layout()
# Plot DSI of percentage of contacts with less than 3 spikes by direction
for cell in df.id.unique():
    DSI = []
    pct_finite_all = []
    for cell in df.id.unique():
        sub_df = df[df.id==cell]
        sub_df['mean_ISI'] = np.isfinite(sub_df['mean_ISI'])
        df_dir = sub_df.groupby('dir_idx')
        pct_finite = df_dir['mean_ISI'].mean()
        theta = df_dir.med_dir.mean()
        DSI.append(varTuning.get_PD_from_hist(theta,pct_finite)[1])
        pct_finite_all.append(np.nanmean(pct_finite))
    df_dsi['DSI_pct_finite'] = DSI
    df_dsi['pct_finite'] = pct_finite_all

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
