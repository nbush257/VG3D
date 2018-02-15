# Plots direction selectivity index stratified on arclength, uses DSI_by_arclength.csv.

from analyze_by_deflection import *

# ============================ #
# edit here #
# ============================ #
save_loc = os.path.join(os.environ['BOX_PATH'],'__VG3D/_deflection_trials/_NEO/results')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['font.sans-serif'] = 'Arial'
dpi_res = 600
fig_width = 6.9 # in
fig_height = 9 # in
ext = 'png'
# ============================= #
# ============================= #
wd = fig_width/3
ht = wd/0.4
df = pd.read_csv(os.path.join(os.environ['BOX_PATH'],'__VG3D/_deflection_trials/_NEO/results/DSI_by_arclength.csv'))


df = df.pivot(index='id',columns='Arclength')
df2 = df[df.theta_pref.Medial.isnull()]
df2 = df2.drop('Medial', level=1,axis=1)
df = df.dropna()
df_norm = df['DSI'].subtract(df['DSI']['Proximal'],axis='rows').sort_values(by='Distal')
df2_norm = df2['DSI'].subtract(df2['DSI']['Proximal'],axis='rows').sort_values(by='Distal')

# 3 Arclength
f = plt.figure(figsize=(wd,ht))
sns.heatmap(df_norm,vmin=-1,vmax=1,cmap='RdBu_r',linewidth=0.2,linecolor=[0.3,0.3,0.3],cbar=False)
plt.yticks([])
plt.ylabel('Cells')
plt.tight_layout()
plt.savefig(os.path.join(save_loc,'arclength_by_DSI_3.{}'.format(ext)),dpi=dpi_res)
plt.close('all')
# 2 Arclength
f = plt.figure(figsize=(wd,ht))
sns.heatmap(df2_norm,vmin=-1,vmax=1,cmap='RdBu_r',linewidth=0.2,linecolor=[0.3,0.3,0.3])
plt.yticks([])
plt.ylabel('Cells')
plt.tight_layout()
plt.draw()
plt.savefig(os.path.join(save_loc,'arclength_by_DSI_2.{}'.format(ext)),dpi=dpi_res)
plt.close('all')

