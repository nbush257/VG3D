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
ext = 'png'
# ============================= #
# ============================= #
wd = fig_width/3
ht = wd
df = pd.read_csv(os.path.join(os.environ['BOX_PATH'],'__VG3D/_deflection_trials/_NEO/results/DSI_by_arclength.csv'))

df = df[df.stim_responsive]
pvt = pd.pivot_table(df,columns='Arclength',index='id',values='theta_pref')
delta_theta =pvt['Distal']-pvt['Proximal']
r,theta = np.histogram(delta_theta,np.arange(-np.pi,np.pi,np.pi/10))
theta = theta[:-1]+np.mean(np.diff(theta))
sns.set_style('ticks')
f = plt.figure(figsize=(wd,ht))
ax = f.add_subplot(111,projection='polar')
plt.polar(theta,r,'k--')
plt.fill_between(theta,np.zeros_like(r),r,color='k',alpha=0.4)
plt.title('$\\hat\\theta_{Distal} - \\hat\\theta_{Proximal}$',y=1.1)
ax.set_rticks(np.arange(0, np.max(r),5))
ax.set_thetagrids(np.arange(0,360,90),frac=1.2)
plt.tight_layout()
plt.savefig(os.path.join(save_loc,'delta_theta_pref.{}'.format(ext)),dpi=dpi_res)
