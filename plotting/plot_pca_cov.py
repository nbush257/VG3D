from analyze_by_deflection import *
# ============================ #
# edit here #
# ============================ #
save_loc = os.path.join(os.environ['BOX_PATH'],r'__VG3D/_deflection_trials/_NEO/results')
cell_list = ['201711B2','201708D1']
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
# =================================
# Plot PCA variance explained
dat = np.load(os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results\cov_exp_var.npz'))
wd = fig_width/3
ht = wd/.75
f = plt.figure(figsize=(wd,ht))
exp_var = dat['exp_var']
plt.plot(exp_var,'k',alpha=.15,linewidth=1)
plt.plot(np.mean(exp_var,axis=1),color='r',linewidth=3)
sns.despine()
plt.ylim(0,1)
plt.yticks([0,.5,1])
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.tight_layout()
plt.savefig(os.path.join(save_loc,'PCA_var_explained.{}'.format(ext)),dpi=dpi_res)
plt.close('all')

# ================================
# Plot Covariance matrices
sns.set_style('white')
wd = fig_width/2
ht=wd
for cell in cell_list:
    f = plt.figure(figsize=(wd,ht))
    idx = np.where(dat['id']==cell)[0][0]
    cov = dat['cov'][:,:,idx]
    mask = np.zeros_like(cov, dtype='bool')
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.color_palette('RdBu_r', 16)
    sns.heatmap(cov, cmap='RdBu_r', vmin=-1, vmax=1, mask=mask)#,linecolor=[0.3,0.3,0.3],linewidths=0.5)
    ax = plt.gca()
    ax.set_facecolor([0.4,0.4,0.4])
    plt.draw()
    plt.xticks(np.arange(0.5,8),dat['var_labels'].tolist())
    plt.yticks(np.arange(7.5,0,-1),dat['var_labels'].tolist(),rotation=0)
    plt.title('Variable covariance {}'.format(cell))
   plt.tight_layout()
    plt.savefig(os.path.join(save_loc,'var_covariance_{}.{}'.format(cell,ext)),dpi=dpi_res)
    plt.close('all')

