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
            if not np.any(temp_idx):
                continue
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

    # get_stim_responsive columns
    stim_responsive_file = os.path.join(p_save,'cell_id_stim_responsive.csv')
    if os.path.isfile(stim_responsive_file):
        is_stim = pd.read_csv(stim_responsive_file)
        DF = DF.merge(is_stim, on='id')
        DF_ALL = DF_ALL.merge(is_stim, on='id')
        DF_DIRECTION = DF_DIRECTION.merge(is_stim, on='id')

    DF.to_csv(os.path.join(p_save,'onset_data.csv'),index=False)
    DF_ALL.to_csv(os.path.join(p_save, 'onset_tuning_by_cell.csv'),index=False)
    DF_DIRECTION.to_csv(os.path.join(p_save, 'onset_tuning_by_cell_and_direction.csv'),index=False)


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

def get_anova_pvals(p_load):
    """
    Takes direction_arclength_FR_group_anova and
    saves a dataframe of pvalues for significant cells only
    :param df:
    :return:
    """
    df = pd.read_csv(os.path.join(p_load,'direction_arclength_FR_group_anova.csv'))
    is_stim = pd.read_csv(os.path.join(p_load,'cell_id_stim_responsive.csv'))
    df = df.merge(is_stim,on='id')
    df= df[df.stim_responsive]
    df_pvt = pd.pivot_table(df,
                            values='PR(>F)',
                            index='id',
                            columns='test')
    df_pvt = df_pvt.rename(index=str,columns={'C(Arclength)':'Arclength',
                                     'C(Arclength):C(Direction)':'Interaction',
                                     'C(Direction)':'Direction'})
    df_pvt = df_pvt[['Arclength','Direction','Interaction']]
    df_pvt.to_csv(os.path.join(p_load,'anova_pvals.csv'))
    print('Saved to {}'.format(os.path.join(p_load,'anova_pvals.csv')))
    return None



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
            t_edges_temp, PSTH_temp, w = spikeAnalysis.get_time_stretched_PSTH(sub_trains,nbins=25)
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

def DSI_by_cell(p_load):
    """
    calculate the directional selectivity for all cells
    collapsing across all other variables
    :param p_load: location in which the input data live
                    inputs data from direction_arclength_FR_data
    :return: None, saves a csv
    """

    # load data in and use only good cells
    df = pd.read_csv(os.path.join(p_load,r'direction_arclength_FR_group_data.csv'))
    is_stim = pd.read_csv(os.path.join(p_load,r'cell_id_stim_responsive.csv'))
    df = df.merge(is_stim,on='id')
    df= df[df.stim_responsive]

    # init population list
    theta_pref = []
    DSI = []
    id_idx=[]
    cell_list = df.id.unique()

    for cell in cell_list:
        # get medians by cell
        sub_df = df[df.id==cell]
        medians = sub_df.groupby('Direction').median()
        theta = medians.med_dir
        FR = medians.Firing_Rate

        # calculate the angular stats and append to population lists
        theta_pref_sub, DSI_sub= varTuning.get_PD_from_hist(theta,FR)
        DSI.append(DSI_sub)
        theta_pref.append(theta_pref_sub)
        id_idx.append(cell)
    # map population lists to dataframe outpu and save
    DF = pd.DataFrame()
    DF['DSI']=DSI
    DF['theta_pref']=theta_pref
    DF['id']=id_idx
    DF = DF.fillna(0)
    DF.to_csv(os.path.join(p_load,'DSI_by_cell.csv'),index=False)
def directional_selectivity_by_arclength(p_load):
    """
    takes a dataframe that has the arclength, direction, and FR data
    :param df:
    :param p_load:
    :return:
    """
    df = pd.read_csv(os.path.join(p_load,r'direction_arclength_FR_group_data.csv'))
    # is_stim = pd.read_csv(os.path.join(p_load,r'cell_id_stim_responsive.csv'))
    # df = df.merge(is_stim,on='id')
    # df= df[df.stim_responsive]
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

    DF_out.to_csv(os.path.join(p_load, 'DSI_by_arclength.csv'), index=False)


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


def get_onset_and_duration_spikes(p_load,dur=10*pq.ms):
    """
    loops through all the data we have and gets the
    number of spikes during an onset duration,
    the total number of spikes during the contact duration,
    and the length of the contact. This will allow us to calculate how much
    the spiking occurs in the first interval

    :param p_load: directory where the h5 files live
    :param dur: a python quantity to determine the 'onset' epoch

    :return: a dataframe with a summary of the relevant data
    """
    df_all = pd.DataFrame()
    for f in glob.glob(os.path.join(p_load,'*.h5')):
        blk = neoUtils.get_blk(f)
        num_units = len(blk.channel_indexes[-1].units)
        for unit_num in range(num_units):
            df = pd.DataFrame()
            id = neoUtils.get_root(blk,unit_num)
            print('Working on {}'.format(id))
            _, _, trains = spikeAnalysis.get_contact_sliced_trains(blk, unit_num)

            dir_idx, med_angle = worldGeometry.get_contact_direction(blk, plot_tgl=False)

            dir = []
            full=[]
            contact_duration=[]
            onset=[]
            for train,direction in zip(trains,dir_idx):
                onset.append(len(train.time_slice(
                    train.t_start,
                    train.t_start+dur)
                ))
                full.append(len(train))
                dir.append(direction)
                contact_duration.append(float(train.t_stop-train.t_start))

            df_dir = pd.DataFrame()
            df_dir['dir_idx'] = dir
            df_dir['time'] = contact_duration
            df_dir['total_spikes'] = full
            df_dir['onset_spikes'] = onset
            df_dir['med_angle'] = [med_angle[x] for x in df_dir.dir_idx]
            df_dir['id'] = id
            df_all = df_all.append(df_dir)
            df_all['onset_period'] = dur
    return(df_all)
def get_adaptation_v2(p_load):
    """
    log(onset_rate/all_rate)
    :param p_load:
    :return:
    """
    df = pd.read_csv(os.path.join(p_load,'onset_spikes_10ms.csv'))
    is_stim = pd.read_csv(os.path.join(p_load,'cell_id_stim_responsive.csv'))
    df = df.merge(is_stim,on='id')
    df = df[df.stim_responsive]
    DF = pd.DataFrame()
    for cell in df.id.unique():
        sub_df = df[df.id==cell]
        subdf_adaptation=pd.DataFrame()
        totals = sub_df.groupby('dir_idx').sum()
        onset_rate = totals.onset_spikes/totals.onset_period
        duration_rate = totals.total_spikes/totals.time
        adaptation = np.log(onset_rate/duration_rate)
        subdf_adaptation['adaptation_index']=adaptation
        subdf_adaptation['id']=cell
        subdf_adaptation['med_angle'] = sub_df.groupby('dir_idx').mean()['med_angle']
        DF = DF.append(subdf_adaptation)

    DF.to_csv(os.path.join(p_load,'adapation_index_10ms_vs_all.csv'),index=False)
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
        adaptation = np.log(means[0].rate)-np.log(means[10].rate)
        adaptation_df = pd.DataFrame()
        adaptation_df['id']=[cell for x in xrange(len(adaptation))]
        adaptation_df['adaptation_index']=adaptation
        adaptation_df['med_dir']=means[0].med_angle
        adaptation_df['dir_idx']=np.arange(8)

        df_all = df_all.append(adaptation_df)
    return(df_all)

def ISI_by_deflection(blk,unit_num=0):
    unit = blk.channel_indexes[-1].units[unit_num]
    ISI = spikeAnalysis.get_contact_sliced_trains(blk, unit)[1]
    CV,LV = spikeAnalysis.get_CV_LV(ISI)
    mean_ISI= np.array([np.mean(x) for x in ISI])
    idx, med_angle = worldGeometry.get_contact_direction(blk, plot_tgl=False)
    df = pd.DataFrame()
    df['id'] = [neoUtils.get_root(blk,unit_num) for x in range(len(ISI))]
    df['mean_ISI'] = mean_ISI
    df['CV'] = CV
    df['LV'] = LV
    df['dir_idx'] = idx
    df['med_dir'] = [med_angle[x] for x in idx]

    return(df)

def batch_ISI_by_deflection(p_load):
    DF = pd.DataFrame()
    for f in glob.glob(os.path.join(p_load,'*.h5')):
        blk = neoUtils.get_blk(f)
        print('Working on {}'.format(os.path.basename(f)))
        num_units = len(blk.channel_indexes[-1].units)
        # _,med_dir = worldGeometry.get_contact_direction(blk,plot_tgl=False)
        for unit_num in xrange(num_units):
            df = ISI_by_deflection(blk,unit_num)
            DF =  DF.append(df)
    return(DF)
