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
        #TODO: probably want to calculate dispersion across only quantites or derivatives,
        #TODO: and maybe want to classify an eigenvector as quantity/derivative
        q = np.sum(X**4,axis=1)
        # non_deriv+derv = 1
        non_deriv = np.sum(X[:,:8]**2,axis=1)
        if plot_tgl:
            plt.hist(non_deriv)
            plt.title('sum of weights on non_deriv eigenvectors')
            plt.ylabel('# of eigenvectors')
            plt.xlabel('Sum of non derivative loadings')

        NON_DERIV.append(non_deriv)
    NON_DERIV = np.concatenate(NON_DERIV)

    bigX = np.array(bigX)
    mX = np.mean(bigX**2,axis=0)

    semX = np.std(bigX**2,axis=0)#/np.sqrt(bigX.shape[0])

    #mean eigenvector loadings across whiskers

    plt.imshow(mX,cmap='gray')
    plt.colorbar()
    plt.yticks(range(16),df.index.unique())
    plt.xticks(range(16),sub_df.drop(meta_list,axis=1).columns,rotation=45)
    plt.title('Mean input space Eigenvector loading squared, all whiskers')

    #std eigenvector loadings across whiskers
    plt.imshow(semX,cmap='gray')
    plt.colorbar()
    plt.yticks(range(16),df.index.unique())
    plt.xticks(range(16),sub_df.drop(meta_list,axis=1).columns,rotation=45)
    plt.title('STD of input space Eigenvector loading squared, all whiskers')


# TODO: This plot is the mean eigen vector loadings across whiskers, but it needs to be edited a little bit more to be seful
    for ii in range(3):
        plt.plot(mX[ii,:])
        plt.fill_between(range(16),mX[ii,:].T-semX[ii,:].T,mX[ii,:]+semX[ii,:],
                         alpha=0.3)


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




