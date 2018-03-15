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
        df.append(sub_df)

    df.to_csv(os.path.join(p_save,'PCA_decompositions.csv'),index=False)
