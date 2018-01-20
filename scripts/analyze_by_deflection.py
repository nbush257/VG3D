import neoUtils
import neo
import worldGeometry
import statsmodels.api as sm
import pyvttbl
import spikeAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import varTuning
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
    df['Firing_Rate'] = FR
    df['Arclength'] = idx_S
    df['Direction'] = idx_dir

    df.dropna()

    formula = 'Firing_Rate ~ C(Arclength) + C(Direction) + C(Arclength):C(Direction)'
    model = ols(formula, df).fit()
    aov_table = anova_lm(model, typ=2)
    return df, aov_table

def plot_anova(df)
    sns.set_style('white')
    arclength_labels = list(set(df['Arclength']))
    arclength_labels.sort(reverse=True)

    # plot all together
    fig,ax = plt.subplots()
    sns.boxplot(x='Direction', y='Firing_Rate',hue='Arclength',data=df,palette='Blues',notch=False,width=0.5)
    ax.set_title('{}'.format(root))
    ax.legend(bbox_to_anchor=(.9, 1.1))
    plt.draw()
    sns.despine(offset=10, trim=True)

    # plot just by direction
    fig, ax = plt.subplots()
    sns.boxplot(x='Direction',y='Firing_Rate',data=df,palette='husl',width=0.6)
    ax.set_title('{}'.format(root))
    sns.despine(offset=10, trim=False)

    #plot just by arclength
    fig, ax = plt.subplots()
    sns.boxplot(x='Arclength', y='Firing_Rate', data=df, palette='Blues',width=0.6)
    ax.set_title('{}'.format(root))
    sns.despine(offset=10, trim=False)

    # Factor Plot
    sns.factorplot(x='Direction',y='Firing_Rate',col='Arclength',data=df,kind='box',width=0.5)

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
        y = mean_by_category[:, arclength]
        y = np.concatenate([y,[y[0]]])
        error = sem_by_category[:,arclength]
        error = np.concatenate([error,[error[0]]])
        # ax.plot(x, y ,alpha=0.1)
        ax.fill_between(x, y - error, y + error,color=cmap[ii])
    ax.legend(arclength_labels,bbox_to_anchor=(1.2, 1.1))

    # plot direction selectivity by arclength
    theta_pref = pd.Series()
    DSI = pd.Series()
    for arclength in arclength_labels:
        idx = mean_by_category[:, arclength].index
        x = med_dir[idx]
        theta_pref[arclength],DSI[arclength] = varTuning.get_PD_from_hist(x,mean_by_category[:,arclength])

    # plot arclength selectivity by direction?



    





