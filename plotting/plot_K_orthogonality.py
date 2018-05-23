import plotVG3D
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


figsize = plotVG3D.set_fig_style()[1]
p_load = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')
p_save = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')
fname = os.path.join(p_load,'MID55ms_weight_dot_products_normed.npy')
ortho_mat = np.load(fname)

mean_orthomat = np.mean(ortho_mat,axis=2)
sem_orthomat = np.std(ortho_mat,axis=2)/np.sqrt(ortho_mat.shape[-1])
std_orthomat = np.std(ortho_mat,axis=2)


wd = figsize[0]/1.5
ht = wd/3
fig,ax = plt.subplots(1,2,figsize=(wd,ht),sharex=True,sharey=True)
cbar_ax = fig.add_axes([0.91,0.3,0.02,0.4])
plt.subplot(121)
plt.imshow(mean_orthomat,vmin=0,vmax=1,cmap='gray')
plt.xticks(range(3),['$K_{}$'.format(x) for x in range(3)])
plt.yticks(range(3),['$K_{}$'.format(x) for x in range(3)])
plt.title('Mean $cos(\\theta)$ Between\nEach Weight vector')

plt.subplot(122)
plt.imshow(std_orthomat,vmin=0,vmax=1,cmap='gray')
plt.xticks(range(3),['$K_{}$'.format(x) for x in range(3)])
plt.yticks(range(3),['$K_{}$'.format(x) for x in range(3)])
plt.title('S.D. $cos(\\theta)$ Between\nEach Weight vector')
plt.colorbar(cax=cbar_ax)
plt.tight_layout()
plt.savefig(os.path.join(p_save,'Orthogonalilty_of_K_vectors_over_all_neurons.pdf'))

# ================================================================
wd = figsize[0]
ht = wd/1.5
plt.figure(figsize=(wd,ht))
df = pd.read_csv(os.path.join(p_load,r'pillow_MID_weights_orthogonalized_55ms.csv'),index_col=0)
X = df.melt(id_vars=['id','var'],value_vars=['Filter_0','Filter_1','Filter_2'],var_name='K')
X.value= X.value.abs()
sns.barplot(x='var',y='value',hue='K',data=X)
plt.title('Average value of |K| for each\nvariable and filter (orthonormalized K)')
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(p_save,'average_K_loadings_orthonormalized.pdf'))
plt.close('all')
# ================================================================
wd = figsize[0]
ht = wd/1.5
plt.figure(figsize=(wd,ht))
df = pd.read_csv(os.path.join(p_load,r'pillow_MID_weights_raw_55ms.csv'),index_col=0)
X = df.melt(id_vars=['id','var'],value_vars=['Filter_0','Filter_1','Filter_2'],var_name='K')
X.value= X.value.abs()
sns.barplot(x='var',y='value',hue='K',data=X)
plt.title('Average value of K for each\nvariable and filter (raw K)')
plt.savefig(os.path.join(p_save,'average_K_loadings_raw.pdf'))
# =================================================================

df = pd.read_csv(os.path.join(p_load,r'pillow_MID_weights_orthogonalized_55ms.csv'),index_col=0)
# ==========================
# K vector participation ratios
# ==========================
wd = figsize[0]/3
ht = figsize[0]/3
df_pvt = pd.pivot_table(data=df,values=['Filter_0','Filter_1','Filter_2'],index=['var','id'])
Q =np.power(df_pvt,4).groupby('id').sum()
cmap = ['b','r','g']

bins = np.arange(0,1,0.05)
plt.figure(figsize=(wd,ht))
for ii in range(3):
    sns.distplot(Q['Filter_{}'.format(ii)],bins=bins,
                 kde=False,
                 hist_kws={'histtype':'stepfilled',
                           'alpha':0.4,
                           'lw':0},
                 color=cmap[ii])
plt.legend(['$K_{}$'.format(x) for x in range(3)])
for ii in range(3):
    sns.distplot(Q['Filter_{}'.format(ii)],bins=bins,
                 kde=False,
                 hist_kws={'histtype':'step',
                           'alpha':1,
                           'lw':2},
                 color=cmap[ii])

plt.xlim(0,1)
plt.xlabel('Participation Ratio Value')
plt.ylabel('Frequency (cells)')
plt.axvline(1/16.,c='k',ls='--')
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(p_save,'Participation_ratios_across_K_vectors.pdf'))

# ==========================
# Plot eigenvectors ratios
# ==========================

wd = figsize[0]/1.5
ht = figsize[0]/1.25
plt.figure(figsize=(wd,ht))
idx = Q['Filter_0'].argsort()
for ii in range(3):
    plt.subplot(1,3,ii+1)
    sub_df = pd.pivot_table(df[['id','var','Filter_{}'.format(ii)]],index='id',columns='var',values=['Filter_{}'.format(ii)])
    sub_df = sub_df.abs()
    X = sub_df.as_matrix()
    sns.heatmap(X[idx,:],
                cbar=ii==2,
                vmin=0,
                vmax=1,
                square=False)
    plt.xticks(np.arange(16)+0.5,sub_df.columns.levels[1],rotation=90)
    plt.yticks([])
    if ii==0:
        plt.ylabel('Cell ID (ordered by participation ratio)')
plt.tight_layout()
plt.savefig(os.path.join(p_save,'All_K_values.pdf'))
