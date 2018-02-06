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
    # Create arclength groups
    if idx_S is -1:
        print('Only one arclength group')
        arclength_labels=['Proximal']
    elif idx_S is -2:
        print('Too few contacts')
        return(-1,-1)
    if np.max(idx_S) == 2:
        arclength_labels = ['Proximal', 'Medial', 'Distal']
    elif np.max(idx_S)==1:
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

    DF.to_csv(os.path.join(p_save,'onset_data.csv'))
    DF_ALL.to_csv(os.path.join(p_save, 'onset_tuning_by_cell.csv'))
    DF_DIRECTION.to_csv(os.path.join(p_save, 'onset_tuning_by_cell_and_direction.csv'))


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
                if df_temp is -1:
                    continue
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
        return (-1,-1,-1,-1)

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

    return(PSTH,t_edges,max_fr,med_angle)


def batch_peak_PSTH_time(p_load,p_save):
    df = pd.DataFrame()
    for f in glob.glob(os.path.join(p_load,'*.h5')):
        blk = neoUtils.get_blk(f)
        print('Working on {}'.format(os.path.basename(f)))
        num_units = len(blk.channel_indexes[-1].units)
        # _,med_dir = worldGeometry.get_contact_direction(blk,plot_tgl=False)
        for unit_num in xrange(num_units):
            id = neoUtils.get_root(blk,unit_num)
            PSTH,t_edges,max_fr,med_dir = get_PSTH_by_dir(blk,unit_num)
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
    """
    takes a dataframe that has the arclength, direction, and FR data
    :param df:
    :param p_save:
    :return:
    """
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


def get_adaptation_df(p_load,max_t=20):
    '''
    Returns a dataframe that has the firing rate for the
    first N (default=20) ms for each cell and direction. Should be useful for
    calculating an 'adaptation' parameter.

    :param p_load: path to where all the neo files exist
    :param max_t: maximum time in miliseconds to grab the firing rate
    :return:
    '''
    df_all = pd.DataFrame()
    for f in glob.glob(os.path.join(p_load,'*.h5')):
        blk = neoUtils.get_blk(f)
        num_units = len(blk.channel_indexes[-1].units)
        for unit_num in range(num_units):
            df = pd.DataFrame()
            id = neoUtils.get_root(blk,unit_num)
            print('Working on {}'.format(id))
            PSTH,edges,_,med_angle = get_PSTH_by_dir(blk,unit_num,norm_dur=False,binsize=1*pq.ms)
            if PSTH is -1:
                continue
            for ii in xrange(len(PSTH)):
                df_dir = pd.DataFrame()
                df_dir['dir_idx'] = np.repeat(ii,max_t)
                df_dir['time'] = edges[ii][:max_t]
                df_dir['rate'] = PSTH[ii][:max_t]
                df_dir['med_angle'] = med_angle[ii]
                df = df.append(df_dir)
                df['id'] = [id for x in range(len(df))]
        df_all = df_all.append(df)
    return(df_all)


def get_threshold_index(p_load):
    '''
    Return a dataframe with a binary telling you if a particular contact ellicited a spike for each cell
    :param p_load: path to the neo files
    :return: a pandas dataframe with all the contacts for all cells. Cannot reshape since every whisker has a dif. number of contacts
    '''
    df_all = pd.DataFrame()
    for f in glob.glob(os.path.join(p_load,'*.h5')):
        blk = neoUtils.get_blk(f)
        num_units = len(blk.channel_indexes[-1].units)
        for unit_num in xrange(num_units):
            df= pd.DataFrame()
            id = neoUtils.get_root(blk,unit_num)
            print('working on {}'.format(id))
            trains = spikeAnalysis.get_contact_sliced_trains(blk,unit_num)[-1]
            dir_idx,med_dir = worldGeometry.get_contact_direction(blk,plot_tgl=False)
            if dir_idx is -1:
                continue
            dir_map = {key: value for (key, value) in enumerate(med_dir)}
            df['id'] = [id for x in xrange(len(trains))]
            df['did_spike'] = [len(x)>0 for x in trains]
            df['dir_idx'] = dir_idx
            df['med_dir'] = df['dir_idx'].map(dir_map)
            df_all = df_all.append(df)
    return(df_all)


def calc_adaptation(df,binsize=10):
    edges=np.arange(0,df.time.max()+1,10)
    df = df[df.stim_responsive]
    cell_list = df.id.unique()
    df_all = pd.DataFrame()
    for cell in cell_list:
        sub_df = df[df.id==cell]
        df_by_dir = pd.pivot_table(sub_df,index='time',columns='dir_idx',values=['rate','med_angle'])
        means = pd.DataFrame([df_by_dir[x:x+binsize].mean() for x in edges]).T
        means.columns=edges
        adaptation = -np.log(means[10].rate/means[0].rate)
        adaptation_df = pd.DataFrame()
        adaptation_df['id']=[cell for x in xrange(len(adaptation))]
        adaptation_df['adaptation_index']=adaptation
        adaptation_df['med_dir']=means[0].med_angle
        adaptation_df['dir_idx']=np.arange(8)

        df_all = df_all.append(adaptation_df)
    return(df_all)


