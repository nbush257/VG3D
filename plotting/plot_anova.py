from analyze_by_deflection import *

# ============================ #
# edit here #
# ============================ #
save_loc = os.path.join(os.environ['BOX_PATH'],r'___hartmann_lab\papers\VG3D\figures\Fig_B')
cell_list = ['201708D1c0'] # pass a list of ids here if you just want some of the plots, otherwise prints and saves all of them
plt.rcParams['pdf.fonttype'] = 'truetype'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['font.sans-serif'] = 'Arial'
dpi_res = 600
fig_width = 6.9 # in
fig_height = 9 # in
# ============================= #
# ============================= #

df_all = pd.read_csv(os.path.join(
    os.environ['BOX_PATH'],
    '__VG3D/_deflection_trials/_NEO/results/direction_arclength_FR_group_data.csv'
))
sns.set_style('ticks')
df_all = df_all[df_all.stim_responsive]

if len(cell_list)==0:
    cell_list = df_all.id.unique()

for id in cell_list:
    print(id)
    df = df_all[df_all.id==id]
    arclength_labels = list(set(df['Arclength']))
    direction_labels = list(set(df['Direction']))
    arclength_labels.sort(reverse=True)
    med_dir = df[['Direction', 'med_dir']].drop_duplicates().sort_values('Direction')['med_dir'].as_matrix()

    # plot interaction of Arclength and direction
    wd = fig_width/1
    ht = wd/2
    fig,ax = plt.subplots(figsize=(wd,ht))
    sns.boxplot(x='Direction', y='Firing_Rate',hue='Arclength',data=df,palette='Blues',notch=False,width=0.5)
    plt.ylabel('Firing Rate')
    # ax.set_title('{}'.format(id))
    ax.legend(bbox_to_anchor=(.9, 1.1))
    plt.draw()
    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    if save_loc is not None:
        plt.savefig(os.path.join(save_loc, '{}_S_dir_box.pdf'.format(id)), dpi=dpi_res)
    plt.close('all')

    # plot just by direction
    wd = fig_width/2
    ht = wd/1.5
    fig, ax = plt.subplots(figsize=(wd,ht))
    sns.boxplot(x='Direction',y='Firing_Rate',data=df,palette='husl',width=0.6,whis=1,fliersize=2)
    ax.set_title('{}'.format(id))
    sns.despine(offset=10, trim=False)
    plt.ylabel('Firing rate (sp/s)')
    plt.tight_layout()
    if save_loc is not None:
        plt.savefig(os.path.join(save_loc, '{}_dir_box.pdf'.format(id)), dpi=dpi_res)
    plt.close('all')

    #plot just by arclength
    wd = fig_width / 3
    ht = wd /0.5
    fig, ax = plt.subplots(figsize=(wd, ht))
    sns.boxplot(x='Arclength', y='Firing_Rate', data=df,order=['Proximal','Medial','Distal'] ,palette='Blues_r',width=0.6,whis=1,fliersize=2)
    # ax.set_title('{}'.format(id))
    plt.ylabel('Firing rate')
    sns.despine(offset=10, trim=False)
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.draw()
    if save_loc is not None:
        plt.savefig(os.path.join(save_loc, '{}_S_box.pdf'.format(id)), dpi=dpi_res)
    plt.close('all')

    # Factor Plot
    # This has some funky positioning issues
    sns.factorplot(x='Direction',y='Firing_Rate',col='Arclength',data=df,kind='box',width=0.5)
    plt.ylabel('Firing rate')
    plt.tight_layout()
    if save_loc is not None:
        plt.savefig(os.path.join(save_loc, '{}_factor.pdf'.format(id)), dpi=dpi_res)
    plt.close('all')

    # Plot polar by arclength
    wd = fig_width / 3
    ht = wd
    f = plt.figure(figsize=(wd, ht))
    ax = f.add_subplot(111,projection='polar')
    mean_by_category = df.groupby(['Direction', 'Arclength'])['Firing_Rate'].mean()
    sem_by_category = df.groupby(['Direction', 'Arclength'])['Firing_Rate'].sem()
    cmap = sns.color_palette('Blues_r', len(arclength_labels))

    for ii,arclength in enumerate(arclength_labels):
        idx = mean_by_category[:,arclength].index
        x = med_dir[idx]
        x = np.concatenate([x,[x[0]]])
        y = mean_by_category[:, arclength].as_matrix()
        y = np.concatenate([y,[y[0]]])
        error = sem_by_category[:,arclength].as_matrix()
        error = np.concatenate([error,[error[0]]])
        # ax.plot(x, y ,'.',alpha=.8,color=cmap[ii])
        ax.fill_between(x, y - error, y + error,color=cmap[ii])
    # ax.legend(arclength_labels,bbox_to_anchor=(1.2, 1.1))
    tick_vals = np.round(np.linspace(0,np.max(mean_by_category),3),-1)
    ax.set_rticks(tick_vals)
    if save_loc is not None:
        plt.savefig(os.path.join(save_loc, '{}_S_polar.pdf'.format(id)), dpi=dpi_res)
    plt.close('all')

    # plot direction selectivity by arclength

    wd = fig_width / 3
    ht = wd / 0.5
    fig, ax = plt.subplots(figsize=(wd, ht))
    theta_pref = pd.Series()
    DSI = pd.Series()
    for arclength in arclength_labels:
      idx = mean_by_category[:, arclength].index
      x = med_dir[idx]
      theta_pref[arclength],DSI[arclength] = varTuning.get_PD_from_hist(x,mean_by_category[:,arclength])
    sns.barplot(x=DSI.index, y=DSI, palette=cmap)
    ax.set_ylim(0,1)
    plt.xticks(rotation=0)
    sns.despine(offset=5)
    plt.ylabel('Direction Selectivity Index\n(1-Circular Variance)')
    # plt.title('Direction selectivity\nby arclength'.format(id))
    plt.xticks(rotation=60)
    plt.tight_layout()
    if save_loc is not None:
        plt.savefig(os.path.join(save_loc, '{}_dir_selectivity_by_S.pdf'.format(id)), dpi=dpi_res)
    plt.close('all')

    # plot arclength selectivity by direction?
    cmap = sns.color_palette('husl',8)
    tuning_by_dir = pd.Series()
    for direction in direction_labels:
      try:
          tuning_by_dir[str(direction)] =  mean_by_category[direction,'Proximal']/mean_by_category[direction,'Distal']
      except:
          tuning_by_dir[str(direction)] = np.nan

    f = plt.figure()
    plt.polar()
    plt.plot(med_dir,tuning_by_dir,'ko')
    ax = plt.gca()
    theta_fill = np.arange(0, 2, 1. / 360) * np.pi
    plt.fill_between(theta_fill,1.,alpha=0.2,color='r')
    plt.fill_between(theta_fill, 1.,ax.get_rmax(), alpha=0.2,color='g')
    ax.spines['polar'].set_visible(False)
    ax.set_title('Arclength tuning\nby direction group {}'.format(id))
    if save_loc is not None:
        plt.savefig(os.path.join(save_loc, '{}_S_selectivity_by_dir.pdf'.format(id)), dpi=dpi_res)
    plt.close('all')
