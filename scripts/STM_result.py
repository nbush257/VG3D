import glob
import pandas as pd
import neoUtils
import numpy as np
import os

def np_to_pd(fname):
    dat = np.load(fname)
    kernel_sizes = dat['kernel_sizes']
    R = dat['R'].item()
    hist = pd.DataFrame(columns=R.keys())
    no_hist = pd.DataFrame(columns=R.keys())

    for model, method_dict in R.iteritems():
        no_hist[model] = method_dict['yhat']
        hist[model] = method_dict['yhat_sim']
    no_hist['id'] = os.path.basename(fname)[:10]
    hist['id'] = os.path.basename(fname)[:10]
    no_hist['kernels'] = kernel_sizes
    hist['kernels'] = kernel_sizes
    return(hist,no_hist)

def batch_to_pd(p_load):
    first=True
    for f in glob.glob(os.path.join(p_load,'*STM*.npz')):
        try:
            print('Working on {}'.format(os.path.basename(f)))
            hist,no_hist = np_to_pd(f)
        except:
            print('Problem at {}'.format(os.path.basename(f)))
            continue

        if first:
            df_hist = hist
            df_no_hist = no_hist
            first=False
        else:
            df_hist = df_hist.append(hist)
            df_no_hist = df_no_hist.append(no_hist)
    return(df_hist,df_no_hist)



p_load = '/projects/p30144/_VG3D/deflections/_NEO/results'
hist, no_hist = batch_to_pd(p_load)

hist.to_csv(os.path.join(p_load,'hist_correlations.csv'),index=False)
no_hist.to_csv(os.path.join(p_load,'no_hist_correlations.csv'),index=False)
