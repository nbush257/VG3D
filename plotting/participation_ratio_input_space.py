import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotVG3D
import os
figsize = plotVG3D.set_fig_style()[1]

def input_participation_ratios():
    """
    This function will caluclate the input space participation
    ratio, that is-- how distributed are the input space loading factors
    :return:
    """
    plot_tgl=False
    p_save = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')
    fname = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results\PCA_decompositions_new_eigennames.csv')

    df = pd.read_csv(fname,index_col=0)

    # extract the whisker information
    df['whisker'] = [x[-2:] for x in df.id]
    df['row'] = [x[-2] for x in df.id]
    df['col'] = [x[-1] for x in df.id]
    df = df.sort_values(['row','col'])
    num_whiskers = len(df.id.unique())
    meta_list = ['ExplainedVarianceRatio','ExplainedVariance','id','whisker','row','col','id']
    NON_DERIV=[]
    bigX=[]
    for whisker in df.id.unique():
        sub_df = df[df.id==whisker]
        X = sub_df.drop(meta_list,axis=1).as_matrix() # rows are the eigenvectors, columns are the dimensions
        bigX.append(X)
        # q is a metric for how distributed the loading on eigenvector u_i is.
        # If the loading is equal, q=1/N (1/16); if all on one dimension, equal to 1
        q = np.sum(X**4,axis=1)
        # non_deriv+derv = 1
        non_deriv = np.sum(X[:,:8]**2,axis=1)
        if plot_tgl:
            plt.hist(non_deriv)
            plt.title('sum of weights on non_deriv eigenvectors')
            plt.ylabel('# of eigenvectors')
            plt.xlabel('Sum of non derivative loadings')

        NON_DERIV.append(non_deriv)
    IS_NON_DERIV = np.greater(np.array(NON_DERIV),0.5)
    NON_DERIV = np.concatenate(NON_DERIV)

    bigX = np.array(bigX)
    mX = np.mean(bigX**2,axis=0)

    semX = np.std(bigX**2,axis=0)/np.sqrt(bigX.shape[0])
# ============================================
    varlist = sub_df.drop(meta_list,axis=1).columns
    #mean eigenvector loadings across whiskers
    plt.figure(figsize=((14,6)))
    plt.subplot(121)
    plt.imshow(mX,cmap='gray')
    plt.colorbar()
    plt.yticks(range(16),df.index.unique(),rotation=25)
    plt.xticks(range(16),varlist,rotation=45)
    plt.title('Mean input space Eigenvector loading squared, all whiskers')

    #std eigenvector loadings across whiskers
    plt.subplot(122)
    plt.imshow(semX,cmap='gray')
    plt.colorbar()
    plt.yticks(range(16),df.index.unique(),rotation=25)
    plt.xticks(range(16),sub_df.drop(meta_list,axis=1).columns,rotation=45)
    plt.title('STD of input space Eigenvector loading squared, all whiskers')
    plt.tight_layout()

# ============================================
    df_3 = df.loc[df.index[:3]]

    wd = figsize[0]/1.5
    ht = figsize[0]/3
    plt.figure(figsize=(wd,ht))
    w=0.25
    for ii in range(3):
        plt.bar(np.arange(16)+ii*w,mX[ii,:],width=w,yerr=semX[ii,:])
    plt.legend(['Eigenvector {}'.format(x) for x in range(1,4)],bbox_to_anchor=(.75,0.85))
#    for ii in range(3):j
#        plt.plot(mX[ii,:],'o-')
#        plt.fill_between(range(16),mX[ii,:].T-semX[ii,:].T,mX[ii,:]+semX[ii,:],
#                         alpha=0.3,
#                         )
    plt.xticks(range(16),sub_df.drop(meta_list,axis=1).columns,rotation=45)
    plt.ylabel('Loading value (max=1)')
    sns.despine()
    plt.tight_layout()

# ============================
    wd = figsize[0]/3
    ht = figsize[0]/3
    plt.figure(figsize=(wd,ht))
    sns.distplot(NON_DERIV,
                 25,
                 kde=False,
                 color='k')
    sns.despine()
    plt.xlabel('$||u||_2$ for all\nNon derivative quantities')
    plt.ylabel('Eigenvector Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(p_save,'eigenvectors_code_derivatives_or_nonderivatives_but_not_both.pdf'))
# ==============================
# participation ratios
# ==============================
# ==============================
# ==============================
# CALCULATE THE PARTICIPATION RATIOS
# calculates both the quantity or derivative subset, as well as the full
# ==============================
# ==============================
df_q = pd.DataFrame()
#loop over whiskers
for ii,(X,label) in enumerate(zip(bigX,IS_NON_DERIV)):
   # loop over eigenvectors
    qsub=[]
    qfull=[]
    rep=[]
    for val,u in zip(label,X):
        # if the eigenvector is a Quantity representations
        if val:
            qsub.append(np.sum(u[:8]**4)/np.sum(u[:8]**2)**2)
            rep.append('Quantity')

        # if the eigenvector is a derivative representations
        else:
            qsub.append(np.sum(u[8:]**4)/np.sum(u[8:]**2)**2)
            rep.append('Deriv')
        # get the full participation ratio (will probably max out at 0.5 because the eigenvectors tend to be derivative or quantity only)
        qfull.append(np.sum(u**4))
    temp_df = pd.DataFrame(columns=['qsub','qfull','representation'])
    temp_df['qsub']=qsub
    temp_df['qfull']=qfull
    temp_df['representation'] = rep
    temp_df['Eigenvector'] = temp_df.index
    temp_df['id'] = df.id.unique()[ii]
    df_q = df_q.append(temp_df)
df_q.to_csv(os.path.join(p_save,'participation_ratios_PCA.csv'),index=False)

# ==============================
# ==============================
# plot the participation ratios
df_q = pd.read_csv(os.path.join(p_save,'participation_ratios_PCA.csv'))
wd = figsize[0]/3
ht = figsize[0]/3
plt.figure(figsize=(wd,ht))

# quantities,edges1 = np.histogram(df_q[df_q.representation=='Quantity'].qsub,25)
# deriv,edges2 = np.histogram(df_q[df_q.representation=='Deriv'].qsub,25)
# edges1 = edges1[:-1]+np.mean(np.diff(edges1))/2.
# edges2 = edges2[:-1]+np.mean(np.diff(edges2))/2.
# plt.plot(edges1,quantities,':',
#          lw=3)
# plt.fill_between(edges1,np.zeros_like(edges1),quantities,
#                  alpha=0.3)
# plt.plot(edges2,deriv,':',
#          lw=3)
# plt.fill_between(edges2,np.zeros_like(edges1),deriv,
#                  alpha=0.3)
quantities = df_q[df_q.representation=='Quantity']
deriv = df_q[df_q.representation=='Deriv']
# plt.hist([quantities,deriv],histtype='bar',alpha=0.4,bins=25,lw=2)
# plt.hist([quantities,deriv],
#          histtype='stepfilled',
#          lw=2,
#          stacked=False,
#          color=['r','b'],
#          bins=25,
#          alpha=0.5)
sns.distplot(quantities.qsub,kde=False)
sns.distplot(deriv.qsub,kde=False)
plt.legend(['Quantities','Derivatives'],bbox_to_anchor=(0.5,0.8))
plt.axvline(1/8.,color='k',ls=':')

plt.xlabel('Participation Ratio of PCA inputs\n(Using only Quantity or Derivative)')
sns.despine()
plt.ylabel('Frequency\n(# eigenvectors)')
plt.ylim(0,np.max(np.concatenate([quantities,deriv])))
plt.xlim(0,1)
plt.tight_layout()
plt.savefig(os.path.join(p_save,'participation_ratios_PCA_hist.pdf'))

