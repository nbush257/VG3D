# Plots tuning to onset variables. These are r values of single linear regressions for all vars
from analyze_by_deflection import *

# ============================ #
# edit here #
# ============================ #
save_loc = os.path.join(os.environ['BOX_PATH'],'__VG3D/_deflection_trials/_NEO/results')
cell_list = ['201704C3c0'] # pass a list of ids here if you just want some of the plots, otherwise prints and saves all of them
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
# ============================= #
# ============================= #
df = pd.read_csv(r'C:\Users\nbush257\Box Sync\__VG3D\_deflection_trials\_NEO\results\adaptation_index_by_dir.csv')
wd = fig_height/3
ht = wd
for cell in cell_list:
    sub_df = df[df.id==cell]
    sub_df = sub_df.sort_values(by='med_dir')
    f = plt.figure(figsize=(wd,ht))

    plt.bar(np.arange(8),sub_df.adaptation_index,width=0.8,color=[0.6,0.6,0.6])
    sns.despine(offset=5)
    plt.grid('on',axis='y')
    plt.ylabel('Adaptation Index\n$-log(\\frac{FR_{10:20ms}}{FR_{0:10ms}})$')
    plt.xlabel('Direction Group')
    plt.tight_layout()
    plt.savefig(os.path.join(save_loc,'adaptation_index_by_dir_{}.{}'.format(cell,ext)),dpi=dpi_res)
    plt.close('all')
# ===============================
# TODO: I don't know how to plot the adaptation result as a summary

# DSI =[]
# for cell in df.id.unique():
#     sub_df = df[df.id==cell]
#     theta = sub_df.med_dir
#     R = sub_df.adaptation_index
#     idx = np.isfinite(R)
#     if not np.any(idx):
#         continue
#     DSI.append(varTuning.get_PD_from_hist(theta[idx],R[idx])[1])
#
# DSI = np.array(DSI)
# sns.distplot(DSI[np.isfinite(DSI)],kde=False)