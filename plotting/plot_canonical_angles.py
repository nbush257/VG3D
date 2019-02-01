import pandas as pd
import get_whisker_PCA
import os
import numpy as np
import matplotlib.pyplot as plt
import plotVG3D
figsize = plotVG3D.set_fig_style()[1]
p_load = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')
p_save = p_load
df = pd.read_csv(os.path.join(p_load,'PCA_decompositions.csv'))
canonical_angles = get_whisker_PCA.pairwise_canonical_angles(3)
whisker_names = df.id.unique()
whisker_names = np.array([x[-2:] for x in whisker_names])
whisker_names.sort()
whisker_unique = np.unique(whisker_names)

unique_idx = [np.where(whisker_names==x)[0][0] for x in whisker_unique]
unique_idx.sort()

f = plt.figure(figsize = (figsize[0],figsize[0]/3))
for ii in range(3):
    ax = f.add_subplot(1,3,ii+1)
    sns.heatmap(canonical_angles[:,:,ii],cmap='Greys',vmin=0,vmax=1,cbar=False,square=True)
    plt.title('{} Canonical Angle'.format(['First','Second','Third'][ii]))
    plt.xticks(unique_idx,whisker_unique,rotation=45)
    plt.yticks(unique_idx,whisker_unique,rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(p_save,'First_3_canonical_angles_with_deriv.pdf'))

# =======================
# plot canonical angle histogram
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
plt.title('Values of angles between\nsubspaces covered by all whiskers')
plt.ylabel('# of pairwise comparisons')
plt.xlabel('$cos(\\theta)$')
plt.legend(['{} Canonical Angle'.format(x) for x in ['First','Second','Third']])
plt.tight_layout()
plt.savefig(os.path.join(p_save,'canonical_angle_histogram_deriv.pdf'))