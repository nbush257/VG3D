import seaborn as sns
import matplotlib.pyplot as plt
import plotVG3D
import pandas as pd
import os
figsize = plotVG3D.set_fig_style()[1]
sort_val = 'mean'# mean, median, or mode
not_used = ['mean','median','mode']
not_used.pop(not_used.index('{}'.format(sort_val)))
p_load = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')
df = pd.read_csv(os.path.join(p_load,'min_smoothing_entropy.csv'))
df = df.set_index('id')
med= df.median(axis=1)
mode = df.mode(axis=1)
mean = df.mean(axis=1)
df['median'] = med
df['mean'] = mean
df['mode'] = mode[0]

df = df.sort_values('{}'.format(sort_val),ascending=False)

# ===================================
# Plot heatmap of minimum entropy by variable
# ===================================
wd = figsize[0]/3
ht = figsize[1]/2
f = plt.figure(figsize=(wd,ht))
cmap = plt.get_cmap('Greens',10)
sns.heatmap(df.drop(not_used,axis=1),cmap=cmap)
plt.xticks(rotation=45)
plt.yticks([])
plt.ylabel('cell (ordered by {} smoothing)'.format(sort_val))
plt.title('Smoothing of derivative\nwhich gives minimum\nentropy of P(R=1|S) ')
plt.tight_layout()
# ====================================
# Plot histogram of mean, median, mode
# ====================================

# ==================================
# Analysis of correlations:
# We correlated the smoothed derivative with the rate smoothed at the same level
# ==================================
df = pd.read_csv(os.path.join(p_load,'derivative_corr_by_smoothing.csv'),index_col=0)
for col in df.columns:
    if col=='id':
        continue
    df= df.rename(columns={col:int(col[:-2])})


MAX_CORR = []
for cell in df['id'].unique():
    sub_df = df[df['id']==cell]
    sub_df.drop('id',axis=1,inplace=True)
    # sns.heatmap(data=sub_df,cmap='RdBu_r',center=0)
    max_corr = sub_df.abs().idxmax(axis=1).as_matrix()
    MAX_CORR.append(max_corr)

MAX_CORR = np.array(MAX_CORR)
m  = scipy.stats.mode(MAX_CORR,axis=1)[0]
