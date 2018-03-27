import glob
import os
import GLM
import sklearn
import pandas as pd
import neoUtils
def get_components(fname):
    ''' Get the PCA comonents given a filename'''
    varlist = ['M', 'F', 'TH', 'PHIE']
    blk = neoUtils.get_blk(fname)
    cbool = neoUtils.get_Cbool(blk)
    root = neoUtils.get_root(blk,0)[:-2]
    X = GLM.create_design_matrix(blk,varlist)
    X = neoUtils.replace_NaNs(X,'pchip')
    X = neoUtils.replace_NaNs(X,'interp')

    scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
    X[cbool,:] = scaler.fit_transform(X[cbool,:])

    pca = sklearn.decomposition.PCA()
    pca.fit_transform(X[cbool,:])

    return(pca,root)

def analyze_first_eigenvector():
    p_load = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')
    fname = os.path.join(p_load,'PCA_decompositions.csv')
    df = pd.read_csv(fname,index_col=0)
    df_leading = df[df.index=='Eigenvector0'][['Mx','My','Mz','Fx','Fy','Fz','Theta','Phi']]
    leading_array = df_leading.as_matrix()
    equal_weight = np.ones([1,8])/8
    angles = [np.arccos(np.dot(equal_weight,leading_array[ii,:]))[0] for ii in range(leading_array.shape[0])]
    id = df.id.unique()
    df_new = pd.DataFrame()
    df_new['angle']=np.rad2deg(angles)
    df_new['id'] = id
    df_new['whisker'] = [x[-2:] for x in id]
    df_new['row'] = [x[-2] for x in id]
    df_new['col'] = [x[-1] for x in id]
    o = df_new.whisker.unique()
    o.sort()

    plt.figure()
    sns.stripplot(x='row', y='angle', data=df_new, hue='col', jitter=True, palette='Blues',
                  order=['A', 'B', 'C', 'D', 'E'],edgecolor='gray',linewidth=.5)
    plt.title('Angle Between Leading Eigenvector and Equal Weighted vector')
    sns.despine(trim=True)
    plt.grid('on',axis='y')

    plt.figure()
    sns.stripplot(x='col', y='angle', data=df_new, hue='row',hue_order=['A','B','C','D','E'] ,jitter=True, palette='Reds',
                  order=['0','1','2','3','4','5'],edgecolor='gray',linewidth=.5)
    plt.title('Angle Between Leading Eigenvector and Equal Weighted vector')
    sns.despine(trim=True)
    plt.grid('on',axis='y')

    plt.figure()
    sns.stripplot(x='whisker', y='angle', data=df_new, jitter=True,
                  order=o,color='gray',linewidth=.5)
    plt.title('Angle Between Leading Eigenvector and Equal Weighted vector')
    sns.despine(trim=True)
    plt.grid('on',axis='y')


def pairwise_first_eigenvector():
    p_load = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')
    fname = os.path.join(p_load,'PCA_decompositions.csv')
    df = pd.read_csv(fname,index_col=0)
    df['whisker'] = [x[-2:] for x in df.id]
    df['row'] = [x[-2] for x in df.id]
    df['col'] = [x[-1] for x in df.id]
    df = df.sort_values(['row','col'])
    df_leading = df[df.index=='Eigenvector0'][['Mx','My','Mz','Fx','Fy','Fz','Theta','Phi']]
    df_id = df[df.index=='Eigenvector0'][['whisker','row','col']]
    leading_array = df_leading.as_matrix()

    pairwise_matrix = np.empty([leading_array.shape[0],leading_array.shape[0]])
    for ii in range(leading_array.shape[0]):
        for jj in range(leading_array.shape[0]):
           pairwise_matrix[ii,jj] = np.abs(np.dot(leading_array[ii,:],leading_array[jj,:]))
    return(pairwise_matrix)

def pairwise_canonical_angles(num_dims=2):
    pass


if __name__=='__main__':
    p_load = r'C:\Users\guru\Box Sync\__VG3D\_deflection_trials\_NEO'
    p_save = r'C:\Users\guru\Box Sync\__VG3D\_deflection_trials\_NEO\results'
    varlist=['Mx','My','Mz','Fx','Fy','Fz','Theta','Phi']
    df = pd.DataFrame()
    for f in glob.glob(os.path.join(p_load,'*.h5')):
        print('Getting PCs for {}'.format(os.path.splitext(os.path.basename(f))[0]))
        pc_decomp,id = get_components(f)
        sub_df = pd.DataFrame()
        for ii,component in enumerate(pc_decomp.components_):
            for jj,var in enumerate(varlist):
                sub_df.loc['Eigenvector{}'.format(ii),var]=component[jj]

        sub_df['ExplainedVarianceRatio'] = pc_decomp.explained_variance_ratio_
        sub_df['ExplainedVariance'] = pc_decomp.explained_variance_
        sub_df['id']=id
        df = df.append(sub_df)

    df.to_csv(os.path.join(p_save,'PCA_decompositions.csv'))
