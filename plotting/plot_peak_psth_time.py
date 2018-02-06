# Plots direction selectivity index stratified on arclength, uses DSI_by_arclength.csv.

from analyze_by_deflection import *

# ============================ #
# edit here #
# ============================ #
save_loc = os.path.join(os.environ['BOX_PATH'],'__VG3D/_deflection_trials/_NEO/results')
cell_list = ['201708D1c0','201605D1c4']
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
sns.set_style('ticks')
# ============================= #
# ============================= #
wd = fig_width/3
ht = wd/0.4
df = pd.read_csv(os.path.join(os.environ['BOX_PATH'],'__VG3D/_deflection_trials/_NEO/results/peak_PSTH_time.csv'))
df = df[df.stim_responsive]
order =df[['id','peak_time']].groupby('id').var().sort_values(by='peak_time')

f = plt.figure(figsize=(wd,ht))
sns.stripplot(x=df.peak_time,data=df,y='id',order=order.index,color='k')
plt.grid('on',axis='x')
plt.xticks(np.arange(0,1.1,0.25))
plt.yticks([])
plt.ylabel('Cells\nordered by variance')
plt.xlabel('Time in contact (Normalized)')
plt.tight_layout()
plt.savefig(os.path.join(save_loc,'PSTH_peak_times.{}'.format(ext)),dpi=dpi_res)
plt.close('all')


# ===============================
wd = fig_width/3
ht = wd
for cell in cell_list:
    sub_df = df[df.id==cell]
    R = sub_df.peak_time.as_matrix()
    theta = sub_df.med_dir.as_matrix()
    f = plt.figure(figsize=(wd,ht))
    plt.polar(theta,R,'ko',alpha=0.5)
    theta = np.concatenate([theta,[theta[0]]])
    R = np.concatenate([R,[R[0]]])
    plt.polar(theta,R,'k',alpha=0.6)
    ax = plt.gca()
    ax.set_rlim(0,1)
    ax.set_rticks([.5,1])
    plt.title('Time of peak firing\nby direction {}'.format(cell))
    plt.tight_layout()
    plt.savefig(os.path.join(save_loc,'Peak_PSTH_time_by_dir_{}.{}'.format(cell,ext)),dpi=dpi_res)
    plt.close('all')

# ===============================
wd = fig_width/2
ht = wd/1.5
DSI=[]
for cell in df.id.unique():
    sub_df = df[df.id==cell]

    R = sub_df.peak_time.as_matrix()
    theta = sub_df.med_dir.as_matrix()

    theta_pref_temp,DSI_temp = varTuning.get_PD_from_hist(theta,R)
    DSI.append(DSI_temp)
f = plt.figure(figsize=(wd,ht))
sns.distplot(DSI,15,kde=False,color='k')
sns.despine()
plt.grid('on',axis='y')
plt.ylabel('Number of cells')
plt.xlabel('1-CircVar of Peak PSTH time')
plt.yticks(np.arange(0,15,5))
plt.tight_layout()
plt.savefig(os.path.join(save_loc,'DSI_of_peak_time_all_cells.{}'.format(ext)),dpi=dpi_res)
