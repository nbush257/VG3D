from analyze_by_deflection import *
import plotVG3D
import os
import seaborn as sns
# ============================ #

# edit here #
# ============================ #
save_loc = os.path.join(os.environ['BOX_PATH'],r'__VG3D/_deflection_trials/_NEO/pillowX')
cell_list = ['201711B2','201708D1']
figsize = plotVG3D.set_fig_style()[1]
sns.set_style('ticks')
# =================================
# Plot PCA variance explained
dat = np.load(os.path.join(save_loc,'input_pca.npy'))
wd = figsize[0]/3
ht = wd/.75
f = plt.figure(figsize=(wd,ht))
exp_var = np.array([x.explained_variance_ratio_ for x in dat]).T
plt.plot(exp_var,'k',alpha=.15,linewidth=1)
plt.plot(np.mean(exp_var,axis=1),'o-',color = [182/255.,0,0],linewidth=2,markeredgecolor='k',markeredgewidth=1)
sns.despine()
plt.ylim(0,1)
plt.yticks([0,.5,1])
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.tight_layout()
plt.savefig(os.path.join(save_loc,'PCA_var_explained.pdf'))
plt.close('all')
# ==================================
# Plot cumulative variance explained
wd = figsize[0]/2
ht = wd/.75
f = plt.figure(figsize=(wd,ht))
cum_exp_var = np.cumsum(exp_var,axis=0)
plt.plot(cum_exp_var,'k',alpha=.15,linewidth=1)
plt.plot(np.mean(cum_exp_var,axis=1),'o-',color = [182/255.,0,0],linewidth=2,markeredgecolor='k',markeredgewidth=1)
sns.despine()
plt.ylim(0,1.1)
plt.yticks([0,.5,1])
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Variance Explained')
plt.axhline(0.95,color='k',ls='--')
plt.tight_layout()
plt.savefig(os.path.join(save_loc,'cumulative_PCA_var_explained.pdf'))
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
    sns.heatmap(cov, cmap=cmap, vmin=-1, vmax=1, mask=mask)#,linecolor=[0.3,0.3,0.3],linewidths=0.5)
    ax = plt.gca()
    ax.set_facecolor([0.4,0.4,0.4])
    plt.draw()
    plt.xticks(np.arange(0.5,8),dat['var_labels'].tolist())
    plt.yticks(np.arange(0.5,8),dat['var_labels'].tolist(),rotation=0)
    plt.title('Variable covariance {}'.format(cell))
    plt.tight_layout()
    plt.savefig(os.path.join(save_loc,'var_covariance_{}.{}'.format(cell,ext)),dpi=dpi_res)
    plt.close('all')


