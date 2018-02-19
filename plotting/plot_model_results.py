import pandas as pd
import seaborn as sns
sns.set()
sns.set_style('ticks')


fname = '/projects/p30144/_VG3D/deflections/_NEO/results/no_hist_correlations.csv'
df = pd.rad_csv(fname)
df2 =df.melt(id_vars=['id','kernels'],value_vars=['full','noM','noF','noR','noD'],var_name='model_type',value_name='Pearson_Correlation')

sns.factorplot(x='model_type',y='Pearson_Correlation',data=df2,
               hue='kernels',
               kind='box',
               palette='Greens')
sns.boxplot(x='model_type',y='Pearson_Correlation',data=df2[df2.kernels==16])
sns.swarmplot(x='model_type',y='Pearson_Correlation',data=df2[df2.kernels==16],color='k')
# TODO plot differences
