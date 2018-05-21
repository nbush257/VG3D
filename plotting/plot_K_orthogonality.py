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

for ii in range(3):
    plt.subplot(1,3,ii+1)
    sub_df = pd.pivot_table(df[['id','var','Filter_{}'.format(ii)]],index='id',columns='var',values=['Filter_{}'.format(ii)])
    sub_df = sub_df.abs()
    # order by participation ratio?
    sns.heatmap(sub_df)
    plt.xticks(rotation=45)
    plt.yticks([])
plt.savefig(os.path.join(p_save,'All_K_values.pdf'))
