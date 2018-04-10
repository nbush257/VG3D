import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import plotVG3D

figsize = plotVG3D.set_fig_style()[1]
p_load = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\pillowX')
p_results = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')

df = pd.read_csv(os.path.join(p_load,r'pillow_MID_canonical_angles.csv'))
is_stim = pd.read_csv(os.path.join(p_results,'cell_id_stim_responsive.csv'))
df = df.merge(is_stim,on='id')
df = df[df.stim_responsive]
# ===========================================
# plot the canonical angles between Pillow vectors and PCA
# ===========================================
wd = figsize[0]/3
ht = figsize[1]/2
f = plt.figure(figsize=(wd,ht))
sns.heatmap(df[['Angle0','Angle1','Angle2']].sort_values('Angle0',ascending=False),
            vmin=0,
            vmax=1,
            square=False)
plt.yticks([])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(p_load,'canonical_angles_between_PCA_and_pillow_weight_vectors.pdf'))
# ===========================================
# plot the canonical angles for a pairwise comparison of the pillow angles
# ===========================================
wd = figsize[0]/4
ht = wd
df = pd.read_csv(os.path.join(p_load,'pillow_MID_weights_orthogonalized.csv'),index_col=0)
df = df.merge(is_stim,on='id')
df = df[df.stim_responsive]

df['whisker'] = [x[-4:-2] for x in df.id]
df['row'] = [x[-4] for x in df.id]
df['col'] = [x[-3] for x in df.id]

df = df.sort_values(['row','col'])

num_whiskers = len(df.id.unique())
canonical_angles = np.empty([num_whiskers,num_whiskers,3])
cell_list = df.id.unique()
# get the whisker names
whisker_names = df.id.unique()
whisker_names = np.array([x[-4:-2] for x in whisker_names])
whisker_names.sort()
whisker_unique = np.unique(whisker_names)
unique_idx = [np.where(whisker_names==x)[0][0] for x in whisker_unique]
unique_idx.sort()
# get the angles
for ii,cell_u in enumerate(cell_list):
    for jj,cell_v in enumerate(cell_list):
        U = df[df.id==cell_u][['Filter_0','Filter_1','Filter_2']].as_matrix()
        V = df[df.id==cell_v][['Filter_0','Filter_1','Filter_2']].as_matrix()

        prod = np.dot(U.T,V)
        canonical_angles[ii,jj,:] = np.linalg.svd(prod)[1]

# plot the angles
wd = figsize[0]
ht = figsize[1]/4
f = plt.figure(figsize=(wd,ht) )
for ii in range(3):
    ax = f.add_subplot(1,3,ii+1)
    sns.heatmap(canonical_angles[:,:,ii],cmap='Greys',vmin=0,vmax=1,cbar=False,square=True)
    plt.title('{} Canonical Angle'.format(['First','Second','Third'][ii]))
    plt.xticks(unique_idx,whisker_unique,rotation=45)
    plt.yticks(unique_idx,whisker_unique,rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(p_load,'canonical_angles_between_neuron_weights.pdf'))

# ===========================================
# plot a histogram of the canonical angles between the weights
# ===========================================
wd = figsize[0]/1.5
ht = wd
f = plt.figure(figsize=(wd,ht))
x,y = np.tril_indices(canonical_angles.shape[0],-1)
first = canonical_angles[x,y,0]
second = canonical_angles[x,y,1]
third = canonical_angles[x,y,2]

sns.distplot(first,kde=False)
sns.distplot(second,kde=False)
sns.distplot(third,kde=False)
sns.despine()
plt.title('Values of angles between\nsubspaces covered by all neurons')
plt.ylabel('# of pairwise comparisons')
plt.xlabel('$cos(\\theta)$')
plt.legend(['{} Canonical Angle'.format(x) for x in ['First','Second','Third']])
plt.tight_layout()
plt.savefig(os.path.join(p_load,'canonical_angles_between_neuron_weights_histogram.pdf'))
