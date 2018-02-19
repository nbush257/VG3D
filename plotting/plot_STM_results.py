import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style('ticks')
# ===================== # 
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['font.sans-serif'] = 'Arial'
dpi_res = 600
fig_width = 6.9 # in
sns.set_style('ticks')
fig_height = 9 # in
ext = 'png'
# ============================= #
# ============================= #
# 
# ========= No Hist =========== #
fname = '/projects/p30144/_VG3D/deflections/_NEO/results/no_hist_correlations.csv'
df = pd.read_csv(fname)
df2 =df.melt(id_vars=['id','kernels'],value_vars=['full','noM','noF','noR','noD'],var_name='model_type',value_name='Pearson_Correlation')
sns.factorplot(x='model_type',y='Pearson_Correlation',data=df2,
               hue='kernels',
               kind='box',
               palette='Greens')
sns.boxplot(x='model_type',y='Pearson_Correlation',data=df2[df2.kernels==16])
sns.swarmplot(x='model_type',y='Pearson_Correlation',data=df2[df2.kernels==16],color='k')
