from analyze_by_deflection import *

# ============================ #
# edit here #
# ============================ #
save_loc = os.path.join(os.environ['BOX_PATH'],'__VG3D/_deflection_trials/_NEO/results')
cell_list = [] # pass a list of ids here if you just want some of the plots, otherwise prints and saves all of them
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
ext='png'
# ============================= #
# ============================= #
wd=fig_width/4
ht = wd/0.3


df = pd.read_csv(os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results\anova_pvals.csv'))
df = df[df.stim_responsive]
df_thresh = df[['Arclength','Direction','Interaction']]<0.05

f = plt.figure(figsize=(wd,ht))
sns.heatmap(df_thresh,cmap=sns.cubehelix_palette(as_cmap=True),linewidth=0.4,linecolor=[0.6,0.6,0.6],cbar=False)
plt.yticks([])
plt.ylabel('Cell')
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig(os.path.join(save_loc,'significant_anova.{}'.format(ext)),dpi=dpi_res)
