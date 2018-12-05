import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from cycler import cycler
from plotVG3D import set_fig_style
figsize,ext = set_fig_style()[1:3]
cmap = mpl.cm.get_cmap('Greys')
import numpy as np
mpl.rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(100,205,5).astype('int')))
## ================================ ##j
## ==== large NLIN comparisons ==== ##
## ================================ ##
fname_large ='K_testing_large_nlin.csv'
fname_small ='K_testing_small_nlin.csv'
sigma=16

p_load = '__VG3D/_deflection_trials/_NEO/results'
p_load = os.path.join(os.environ['BOX_PATH'],p_load)
p_save = p_load
df1 = pd.read_csv(os.path.join(p_load,fname_large))
df2 = pd.read_csv(os.path.join(p_load,fname_small))

for cell in df1.id.unique():
    sub_df1 = df1[df1.id==cell]

    sub_df1.plot(x='sigma',
             linewidth=1.5,
             marker='o',)

    plt.grid()
    plt.ylim(0,1)
    plt.ylabel('Pearson Correlation (R)')
    plt.xlabel('$\sigma$ (ms)')
    sns.despine()
    plt.title(cell)
    plt.savefig(os.path.join(p_save,'{}_K_test.pdf'.format(cell)))
    plt.close()




df1_16 = df1.loc[df1['sigma'] == sigma]
df2_16 = df2.loc[df2['sigma'] == sigma]

df1_16=df1_16.set_index('id')
df2_16=df2_16.set_index('id')

df1_16.drop('sigma',axis=1,inplace=True)
df2_16.drop('sigma',axis=1,inplace=True)
df1_16 = df1_16.melt(var_name='K_size',value_name='Pearson')
df2_16 = df2_16.melt(var_name='K_size',value_name='Pearson')

df1_16['nonlinearity'] = 'large'
df2_16['nonlinearity'] = 'small'

df_all=pd.concat([df1_16,df2_16],axis=0)

sns.factorplot(x='K_size',y='Pearson',hue='nonlinearity',data=df_all,palette='Set2',marker='X')
plt.ylim(0,1)
plt.savefig(os.path.join(p_save,'K_testing_summary_sigma-{}.pdf'.format(sigma)))

plt.close()