import neoUtils
import neo
import scipy.stats as stats
import worldGeometry
import quantities as pq
import spikeAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import varTuning
import glob
import numpy as np
import os
import matplotlib.ticker as ticker

sns.set()

def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq'] / sum(aov['sum_sq'])
    return aov


def omega_squared(aov):
    mse = aov['sum_sq'][-1] / aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq'] - (aov[:-1]['df'] * mse)) / (sum(aov['sum_sq']) + mse)
    return aov


def anova_analysis(blk,unit_num=0):
    use_flags = neoUtils.concatenate_epochs(blk)
    root = neoUtils.get_root(blk,unit_num)
    idx_dir,med_dir = worldGeometry.get_contact_direction(blk,plot_tgl=False)
    FR = spikeAnalysis.get_contact_sliced_trains(blk,unit_num)[0].magnitude
    idx_S = worldGeometry.get_radial_distance_group(blk,plot_tgl=False)
    if np.max(idx_S) == 2:
        arclength_labels = ['Proximal', 'Medial', 'Distal']
    else:
        arclength_labels = ['Proximal', 'Distal']
    idx_S = [arclength_labels[x] for x in idx_S]
    df = pd.DataFrame()
    directions = pd.DataFrame()
    df['Firing_Rate'] = FR
    df['Arclength'] = idx_S
    df['Direction'] = idx_dir
    df['id'] = root

    directions['med_dir']=med_dir
    directions['Direction']=list(set(df['Direction']))
    df = df.merge(directions)

    df.dropna()

    formula = 'Firing_Rate ~ C(Direction) + C(Arclength) + C(Arclength):C(Direction)'
    model = ols(formula, df).fit(missing='drop')
    aov_table = anova_lm(model, typ=1)
    aov_table['id'] = root

    return df, aov_table

def plot_anova(df,save_loc=None):
    sns.set_style('white')
    arclength_labels = list(set(df['Arclength']))
    direction_labels = list(set(df['Direction']))
    arclength_labels.sort(reverse=True)
    id = df['id'][0]
    med_dir = df[['Direction', 'med_dir']].drop_duplicates().sort_values('Direction')['med_dir'].as_matrix()
    fig,ax = plt.subplots()

    sns.boxplot(x='Direction', y='Firing_Rate',hue='Arclength',data=df,palette='Blues',notch=False,width=0.5)
    ax.set_title('{}'.format(id))
    ax.legend(bbox_to_anchor=(.9, 1.1))
    plt.draw()
    sns.despine(offset=10, trim=True)
    if save_loc is not None:
        plt.savefig(os.path.join(save_loc, '{}_S_dir_box.png'.format(id)), dpi=300)

    # plot just by direction
    fig, ax = plt.subplots()
    sns.boxplot(x='Direction',y='Firing_Rate',data=df,palette='husl',width=0.6)
    ax.set_title('{}'.format(id))
    sns.despine(offset=10, trim=False)
    if save_loc is not None:
        plt.savefig(os.path.join(save_loc, '{}_dir_box.png'.format(id)), dpi=300)

    #plot just by arclength
    fig, ax = plt.subplots()
    sns.boxplot(x='Arclength', y='Firing_Rate', data=df, palette='Blues',width=0.6)
    ax.set_title('{}'.format(id))
    sns.despine(offset=10, trim=False)
    if save_loc is not None:
        plt.savefig(os.path.join(save_loc, '{}_S_box.png'.format(id)), dpi=300)

    # Factor Plot
    sns.factorplot(x='Direction',y='Firing_Rate',col='Arclength',data=df,kind='box',width=0.5)
    if save_loc is not None:
        plt.savefig(os.path.join(save_loc, '{}_factor.png'.format(id)), dpi=300)

    # Plot polar by arclength
    f = plt.figure()
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
        # ax.plot(x, y ,alpha=0.1)
        ax.fill_between(x, y - error, y + error,color=cmap[ii])
    ax.legend(arclength_labels,bbox_to_anchor=(1.2, 1.1))
    if save_loc is not None:
        plt.savefig(os.path.join(save_loc, '{}_S_polar.png'.format(id)), dpi=300)

    # plot direction selectivity by arclength
    theta_pref = pd.Series()
    DSI = pd.Series()
    for arclength in arclength_labels:
        idx = mean_by_category[:, arclength].index
        x = med_dir[idx]
        theta_pref[arclength],DSI[arclength] = varTuning.get_PD_from_hist(x,mean_by_category[:,arclength])
    if save_loc is not None:
        plt.savefig(os.path.join(save_loc, '{}_dir_selectivity_by_S.png'.format(id)), dpi=300)

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
        plt.savefig(os.path.join(save_loc, '{}_S_selectivity_by_dir.png'.format(id)), dpi=300)


def onset_tuning(blk,unit_num=0,use_zeros=True):
    '''
    Calculate the onset velocity in both terms of CP and in terms of rotation.
    Calculate the relationship between the onset firing rate and the different velcocities
    
    :param blk: 
    :param unit_num: 
    :param use_zeros: 
    :return V_cp_fit,V_rot_fit: 
    '''
    use_flags = neoUtils.concatenate_epochs(blk)
    trains = spikeAnalysis.get_contact_sliced_trains(blk,unit_num)[-1]
    apex = neoUtils.get_contact_apex_idx(blk)*pq.ms
    apex_idx = apex.magnitude.astype('int')
    id = neoUtils.get_root(blk,unit_num)
    # get MB and FB at apex
    M = neoUtils.get_var(blk)
    MB = neoUtils.get_MB_MD(M)[0]
    MB_contacts = neoUtils.get_analog_contact_slices(MB,use_flags)
    MB_apex = neoUtils.get_value_at_idx(MB_contacts,apex_idx).squeeze()
    MB_dot = MB_apex/apex

    F = neoUtils.get_var(blk,'F')
    FB = neoUtils.get_MB_MD(F)[0]
    FB_contacts = neoUtils.get_analog_contact_slices(FB, use_flags)
    FB_apex = neoUtils.get_value_at_idx(FB_contacts, apex_idx).squeeze()
    FB_dot = FB_apex / apex
    # Get onset FR
    onset_counts = np.array([len(train.time_slice(train.t_start,train.t_start+dur)) for
                    train,dur in zip(trains,apex)])
    onset_FR = np.divide(onset_counts,apex)
    onset_FR.units=1/pq.s


    # get V_onset_rot
    V_rot,_,D = worldGeometry.get_onset_velocity(blk)

    dir_idx,dir_angle = worldGeometry.get_contact_direction(blk, False)
    if dir_idx is -1:
        return(-1,-1,-1)
    df = pd.DataFrame()
    df['id'] = [id for x in xrange(MB_dot.shape[0])]
    df['MB'] = MB_apex
    df['MB_dot'] = MB_dot
    df['FB_dot'] = FB_dot
    df['FB'] = FB_apex
    df['rot'] = D
    df['rot_dot'] = V_rot

    df['dir_idx'] = dir_idx
    df['FR'] = onset_FR
    df['dir_angle'] = [dir_angle[x] for x in dir_idx]
    df = df.replace(np.inf,np.nan)
    df = df.dropna()

    # FIT:
    fits_all = pd.DataFrame(columns=['id','var','rvalue','pvalue','slope','intercept'])
    fits_direction = pd.DataFrame()
    idx=0
    idx2=0
    for var in ['MB','MB_dot','FB','FB_dot','rot','rot_dot']:
        fit = stats.linregress(df[var], df['FR'])._asdict()
        fits_all.loc[idx, 'id'] = id
        fits_all.loc[idx,'var']=var

        for k,v in fit.iteritems():
            fits_all.loc[idx,k] = v
        idx+=1

        for direction in xrange(np.max(dir_idx)+1):
            temp_idx = df['dir_idx'] == direction
            fit = stats.linregress(df[var][temp_idx], df['FR'][temp_idx])._asdict()
            fits_direction.loc[idx2, 'id'] = id
            fits_direction.loc[idx2, 'var'] = var
            fits_direction.loc[idx2, 'dir_idx'] = direction
            fits_direction.loc[idx2, 'med_dir'] = dir_angle[direction]
            for k,v in fit.iteritems():
                fits_direction.loc[idx2,k] = v
            idx2+=1
    return(fits_all,fits_direction,df)


def batch_onset_tunings(p_load,p_save):
    DF_ALL = pd.DataFrame()
    DF_DIRECTION = pd.DataFrame()
    DF = pd.DataFrame()
    for f in glob.glob(os.path.join(p_load,'*.h5')):
        print('Working on {}'.format(os.path.basename(f)))
        blk = neoUtils.get_blk(f)
        num_units = len(blk.channel_indexes[-1].units)
        for unit_num in xrange(num_units):
            df_all,df_direction,df = onset_tuning(blk,unit_num=unit_num)
            if df_all is -1:
                continue
            DF  = DF.append(df)
            DF_ALL = DF_ALL.append(df_all)
            DF_DIRECTION = DF_DIRECTION.append(df_direction)

    # DF.to_csv(os.path.join(p_save,'onset_data.csv'))
    DF_ALL.to_csv(os.path.join(p_save, 'onset_tuning_by_cell.csv'))
    # DF_DIRECTION.to_csv(os.path.join(p_save, 'onset_tuning_by_cell_and_direction.csv'))

def batch_anova(p_load,p_save):
    '''
    Calculate the anova tables and data by deflection direction and arclength
    :param p_load: 
    :param p_save: 
    :return: 
    '''
    aov = pd.DataFrame()
    df = pd.DataFrame()
    for f in glob.glob(os.path.join(p_load,'*.h5')):
        blk = neoUtils.get_blk(f)
        print('Working on {}'.format(os.path.basename(f)))
        num_units = len(blk.channel_indexes[-1].units)

        for ii in xrange(num_units):
            try:
                df_temp,aov_temp = anova_analysis(blk,unit_num=ii)
                df = df.append(df_temp)
                aov = aov.append(aov_temp)
                # plot_anova(df_temp,save_loc=p_save)
            except:
                print('Problem with {}c{}'.format(os.path.basename(f),ii))

        plt.close('all')
    df.to_csv(os.path.join(p_save,'direction_arclength_FR_group_data.csv'))
    aov.to_csv(os.path.join(p_save, 'direction_arclength_FR_group_anova.csv'))


def get_PSTH_by_dir(blk,unit_num=0,norm_dur=True,binsize=5*pq.ms):
    '''
    Gets the PSTHs for each direction.
    :param blk: 
    :param unit_num: 
    :param norm_dur: 
    :param binsize: 
    :return PSTH, t_edges, max_fr: The PSTH binheights, the bin edges, and the max value of FR  
    '''
    unit = blk.channel_indexes[-1].units[unit_num]
    _, _, trains = spikeAnalysis.get_contact_sliced_trains(blk, unit)

    b, durations = spikeAnalysis.get_binary_trains(trains)

    idx, med_angle = worldGeometry.get_contact_direction(blk, plot_tgl=False)
    if idx is -1:
        return (-1,-1,-1)

    th_contacts, ph_contacts = worldGeometry.get_delta_angle(blk)
    PSTH = []
    t_edges = []
    max_fr = []
    for dir in np.arange(np.max(idx) + 1):
        sub_idx = np.where(idx == dir)[0]
        sub_trains = [trains[ii] for ii in sub_idx]
        if norm_dur:
            t_edges_temp, PSTH_temp, w = spikeAnalysis.get_time_stretched_PSTH(sub_trains)
        else:
            spt = spikeAnalysis.trains2times(sub_trains, concat_tgl=True)
            PSTH_temp, t_edges_temp = np.histogram(spt, bins=np.arange(0, 500, float(binsize)))
            PSTH_temp = PSTH_temp.astype('f8') / len(durations) / pq.ms * 1000.
            w = binsize

        max_fr.append(np.max(PSTH_temp))
        PSTH.append(PSTH_temp)
        t_edges.append(t_edges_temp)
    max_fr = np.max(max_fr)

    return(PSTH,t_edges,max_fr)

def plot_onset_tunings(df_by_cell,df_by_direction,p_save,save_tgl=True):
    cmap = sns.color_palette('Paired',6)
    df_by_cell = df_by_cell[df_by_cell.stim_responsive]
    df_by_direction = df_by_directioplt.pie(aov_results['significant_arclength']'[.mean())n[df_by_direction.stim_respoisumcell_idx = df_by_cell['id'].unique(),
    dfr = df_by_cell[['id','var','rvalue']]
    is_sig = df_by_cell['pvalue']<0.05
    dfr.loc[np.invert(is_sig),'rvalue']=0
    dfr_pvt = dfr.pivot_table('rvalue',['id','var'])
    dfr_pvt = dfr_pvt.unstack()
    sns.heatmap(dfr_pvt,vmin=-1.,vmax=1.,cmap=sns.color_palette('RdBu_r',128))
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.yticks(rotation=0)
    plt.xticks(rotation=60)
    plt.draw()

    # df_by_dir must be the reshaped
    df_tunings = pd.DataFrame(columns=['id','var','theta_k','DSI'])
    for cell in cell_idx:
        f = plt.figure()
        ax = f.add_subplot(111,projection='polar')
        sub_cell = df_by_direction.loc[cell]
        if not np.any(sub_cell['stim_responsive']):
            continue
        varnames = sub_cell.index.get_level_values(0).unique()
        for ii,var in enumerate(varnames):
            R = sub_cell.loc[var].rvalue.abs()
            theta = sub_cell.loc[var].med_dir
            theta_k,DSI = varTuning.get_PD_from_hist(theta,R)
            df_tunings = df_tunings.append(pd.Series([cell,var,theta_k,DSI],index=['id','var','theta_k','DSI']),ignore_index=True)

            ax.annotate('',
                        xy=(theta_k, DSI ),
                        xytext=(0, 0),
                        arrowprops={'arrowstyle': 'simple,head_width=1', 'linewidth': 1, 'color': cmap[ii],'alpha':0.5})
            ax = plt.gca()
            ax.set_rlim(0,1)
            ax.set_title('{}'.format(cell))

        if save_tgl:
            plt.savefig(os.path.join(p_save,'{}_onset_tuning.png'.format(cell)),dpi=300)
        plt.close('all')
        df_tunings.to_csv(os.path.join(p_save,'onset_tuning_direction_strength.csv'))

def batch_peak_PSTH_time(p_load,p_save):
    df = pd.DataFrame()
    for f in glob.glob(os.path.join(p_load,'*.h5')):
        blk = neoUtils.get_blk(f)
        print('Working on {}'.format(os.path.basename(f)))
        num_units = len(blk.channel_indexes[-1].units)
        _,med_dir = worldGeometry.get_contact_direction(blk,plot_tgl=False)
        for unit_num in xrange(num_units):
            id = neoUtils.get_root(blk,unit_num)
            PSTH,t_edges,max_fr = get_PSTH_by_dir(blk,unit_num)
            if PSTH is -1:
                continue
            peak_time = [t_edges[x][np.nanargmax(PSTH[x])] for x in
                        xrange(len(PSTH))]
            df_temp = pd.DataFrame()
            df_temp['id'] =[id for x in range(len(med_dir))]
            df_temp['med_dir'] =med_dir
            df_temp['peak_time'] =peak_time
            df = df.append(df_temp)
    df.to_csv(os.path.join(p_save,'peak_PSTH_time.csv'))
    print('done')

def directional_selectivity_by_arclength(df,p_save):
    theta_pref = []
    DSI = []
    arclength_idx=[]
    id_idx=[]
    cell_list = df.id.unique()
    for cell in cell_list:
        sub_df = df[df.id==cell]
        arclength_labels = sub_df.Arclength.unique()
        sub_df = pd.pivot_table(sub_df,index=['Arclength','Direction'])
        for arclength in arclength_labels:
            theta = sub_df.loc[arclength].med_dir
            R = sub_df.loc[arclength].Firing_Rate
            theta_pref_temp,DSI_temp = varTuning.get_PD_from_hist(theta,R)
            theta_pref.append(theta_pref_temp)
            DSI.append(DSI_temp)
            arclength_idx.append(arclength)
            id_idx.append(cell)
    DF_out = pd.DataFrame()
    DF_out['id'] = id_idx
    DF_out['Arclength'] = arclength_idx
    DF_out['theta_pref'] = theta_pref
    DF_out['DSI'] = DSI
    DF_out.to_csv(os.path.join(p_save,'DSI_by_arclength.csv'))


if __name__=='__main__':
    p_load = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO')
    p_save = os.path.join(os.environ['BOX_PATH'], r'__VG3D\_deflection_trials\_NEO\results')

    batch_anova(p_load, p_save)

