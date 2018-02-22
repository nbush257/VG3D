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
cell_list = ['201708D1c0']
dpi_res = 600
fig_width = 6.9 # in
sns.set_style('ticks')
fig_height = 9 # in
ext = 'png'
p_save = r'C:\Users\guru\Box Sync\__VG3D\_deflection_trials\_NEO\results'
df = pd.read_csv(os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results\regularity_by_contact.csv'))
df = df[df.stim_responsive]
# =======================================
if len(cell_list)==0:
    cell_list = df.id.unique()
# Plot regularity by direction for a cell
for cell in cell_list:
    sub_df =df[df.id==cell]
    df_dir = sub_df.groupby('dir_idx')
    theta = df_dir.med_dir.mean()
    wd = fig_width/3
    ht = wd
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
