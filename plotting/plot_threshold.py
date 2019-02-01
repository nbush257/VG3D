from analyze_by_deflection import *

# ============================ #
# edit here #
# ============================ #
save_loc = os.path.join(os.environ['BOX_PATH'],'__VG3D/_deflection_trials/_NEO/results')
cell_list = [] # pass a list of ids here if you just want some of the plots, otherwise prints and saves all of them
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
# ============================= #
# ============================= #
df = pd.read_csv(r'C:\Users\nbush257\Box Sync\__VG3D\_deflection_trials\_NEO\results\threshold_binary.csv')
df = df[df.stim_responsive]

pct_spike = df.groupby(['id','dir_idx']).mean()
DSI=[]
for cell in df.id.unique():
    R = pct_spike.loc[cell].did_spike
    theta = pct_spike.loc[cell].med_dir
    DSI.append(varTuning.get_PD_from_hist(theta,R)[1])
sns.distplot(DSI,20,kde=False)
