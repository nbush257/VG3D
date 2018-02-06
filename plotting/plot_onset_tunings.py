# Plots tuning to onset variables. These are r values of single linear regressions for all vars
from analyze_by_deflection import *

# ============================ #
# edit here #
# ============================ #
save_loc = os.path.join(os.environ['BOX_PATH'],'__VG3D/_deflection_trials/_NEO/results')
cell_list = ['201708D1c0'] # pass a list of ids here if you just want some of the plots, otherwise prints and saves all of them
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
ht = wd/0.2
category_labels = ['FB','$\\dot{FB}$','MB','$\\dot{MB}$','ROT','$\\dot{ROT}$']

df_by_cell = pd.read_csv(os.path.join(os.environ['BOX_PATH'],'__VG3D/_deflection_trials/_NEO/results/onset_tuning_by_cell.csv'))
df_by_direction = pd.read_csv(os.path.join(os.environ['BOX_PATH'],'__VG3D/_deflection_trials/_NEO/results/onset_tuning_by_cell_and_direction.csv'))


cmap = sns.color_palette('Paired',6)
df_by_cell = df_by_cell[df_by_cell.stim_responsive]
df_by_direction = df_by_direction[df_by_direction.stim_responsive]
if len(cell_list)==0:
    cell_list = df_by_cell['id'].unique()

dfr = df_by_cell[['id','var','rvalue']]
is_sig = df_by_cell['pvalue']<0.05
dfr.loc[np.invert(is_sig),'rvalue']=0
dfr_pvt = dfr.pivot_table('rvalue',['id','var'])
dfr_pvt = dfr_pvt.unstack()
aa = np.array(dfr_pvt)
idx = np.mean(aa,axis=1).argsort()[::-1]

# plot summary for all cells
f = plt.figure(figsize=(wd,ht))
sns.heatmap(aa[idx,:],vmin=-1.,vmax=1.,cmap=sns.color_palette('RdBu_r',256))
ax = plt.gca()
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
plt.yticks([])
plt.ylabel('Cell',rotation=0,labelpad=10)
locs = plt.xticks()[0]
plt.xticks(locs,category_labels,rotation=60)
plt.xlabel('')
plt.draw()
plt.tight_layout()
plt.savefig(os.path.join(save_loc,'onset_tuning_all.{}'.format(ext)),dpi=dpi_res)

# df_by_dir must be the reshaped
df_tunings = pd.DataFrame(columns=['id','var','theta_k','DSI'])
sns.set_style('ticks')
wd=fig_width/2
ht=wd
for cell in cell_list:
    f = plt.figure(figsize = (wd,ht))
    ax = f.add_subplot(111,projection='polar')
    sub_cell = df_by_direction[df_by_direction.id==cell]
    if not np.any(sub_cell['stim_responsive']):
        continue
    varnames = sub_cell['var'].unique()
    for ii,var in enumerate(varnames):
        R = sub_cell[sub_cell['var']==var].rvalue.abs()
        theta = sub_cell[sub_cell['var']==var].med_dir.abs()
        theta_k,DSI = varTuning.get_PD_from_hist(theta,R)
        df_tunings = df_tunings.append(pd.Series([cell,var,theta_k,DSI],index=['id','var','theta_k','DSI']),ignore_index=True)
        plt.plot(0,0,'o',color=cmap[ii])
        ax.annotate('',
                    xy=(theta_k, DSI ),
                    xytext=(0, 0),
                    arrowprops={'arrowstyle': 'simple,head_width=1', 'linewidth': 1, 'color': cmap[ii],'alpha':0.6})
        ax = plt.gca()
        ax.set_rlim(0,1)
        ax.set_rticks([0,.5,1])

        ax.set_title('Onset tuning\n{}'.format(cell))
        # ax.legend(varnames,bbox_to_anchor=(1.15,1.15))
    plt.tight_layout()

    plt.savefig(os.path.join(save_loc,'{}_onset_tuning.{}'.format(cell,ext)),dpi=dpi_res)
    plt.close('all')

