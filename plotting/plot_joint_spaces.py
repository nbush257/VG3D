from analyze_by_deflection import *
import sklearn
import pandas as pd
import GLM
import varTuning
import quantities as pq
import matplotlib.ticker as mtick
# ============================ #
# edit here #
# ============================ #
save_loc = os.path.join(os.environ['BOX_PATH'],'__VG3D/_deflection_trials/_NEO/results')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['font.sans-serif'] = 'Arial'
dpi_res = 600
fig_width = 6.9 # in
fig_height = 9 # in
ext = 'png'
# ============================= #
# ============================= #
# spaces to plot: MyMZ, FyFz,THPHI,MxFx

def nl(x,k_scale=80):
    k = 1/np.percentile(x,k_scale)
    z = 2/(1+np.exp(-k*x)) - 1
    invnl = lambda y: -1/k*(np.log(1-y)-np.log(1+y))
    return z,invnl

def plot_single_joint_space(var1,var2,r,cbool,bins,bin_stretch=True,ax=None):

    idx = np.logical_and(np.isfinite(var1),np.isfinite(var2)).ravel()
    if bin_stretch:
        var1s = np.empty(var1.shape[0])
        var2s = np.empty(var2.shape[0])
        var1s[idx], logit_y = nl(var1[idx], 90)
        var2s[idx], logit_z = nl(var2[idx], 90)
    else:
        var1s = var1
        var2s = var2

    response, var1_edges, var2_edges = varTuning.joint_response_hist(var1s, var2s, r, cbool, bins=bins, min_obs=10)
    if bin_stretch:
        var1_edges = logit_y(var1_edges)
        var2_edges = logit_z(var2_edges)
    else:
        pass

    h= plt.pcolormesh(var1_edges[:-1], var2_edges[:-1], response, cmap='OrRd')
    if ax is None:
        ax = plt.gca()

    ax.axvline(color='k', linewidth=1)
    ax.axhline(color='k', linewidth=1)
    ax.patch.set_color([0.6, 0.6, 0.6])

    mask = response.mask.__invert__()
    if not mask.all():
        ax.set_ylim(var2_edges[np.where(mask)[0].min()], var2_edges[np.where(mask)[0].max()])
        ax.set_xlim(var1_edges[np.where(mask)[1].min()], var1_edges[np.where(mask)[1].max()])
    return(h)


def plot_joint_spaces(blk,unit_num=0,bin_stretch=False,p_save=None):
    id = neoUtils.get_root(blk,unit_num)
    use_flags = neoUtils.concatenate_epochs(blk)
    cbool = neoUtils.get_Cbool(blk)
    r,b = neoUtils.get_rate_b(blk,unit_num,2*pq.ms)

    M = neoUtils.get_var(blk, 'M').magnitude
    F = neoUtils.get_var(blk, 'F').magnitude
    TH = neoUtils.get_var(blk, 'TH').magnitude
    PH = neoUtils.get_var(blk, 'PHIE').magnitude

    TH = neoUtils.center_var(TH,use_flags)
    PH = neoUtils.center_var(PH, use_flags)

    TH[np.invert(cbool)] = np.nan
    PH[np.invert(cbool)] = np.nan

    vars = {}
    names = {}
    vars[0] = [M[:, 1].ravel(), M[:, 2].ravel()]
    vars[1] = [F[:, 1].ravel(), F[:, 2].ravel()]
    vars[2] = [M[:, 0].ravel(), F[:, 0].ravel()]
    vars[3] = [TH.ravel(), PH.ravel()]

    names[0] = ['M_y', 'M_z']
    names[1] = ['F_y', 'F_z']
    names[2] = ['M_x', 'F_x']
    names[3] = ['\\Delta\\theta', '\\Delta\\phi']
    titles = ['Bending','Lateral','Axial','Rotation']
    f = plt.figure()
    handles=[]
    for ii,var in vars.iteritems():
        ax = f.add_subplot(2,2,ii+1)
        handles.append(plot_single_joint_space(var[0],var[1],r,cbool,50,ax=ax,bin_stretch=False))
        ax.set_xlabel('${}$'.format(names[ii][0]))
        ax.set_ylabel('${}$'.format(names[ii][1]))

        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
        plt.xticks(rotation=25)
        ax.set_title(titles[ii])


        if ii==3:
            ax.axis('equal')
            max_color = np.max([x.get_clim() for x in handles])

    # set all colors to same scale
    for h in handles:
        h.set_clim(0,max_color)

    plt.colorbar(handles[0])
    plt.suptitle('{}'.format(id))
    plt.tight_layout()
    if p_save is not None:
        plt.savefig(os.path.join(p_save,'{}_joint_spaces.png'.format(id)),dpi=dpi_res)
        plt.close('all')
    return(0)


def plot_pca_spaces(fname,unit_num,p_smooth=None,deriv_smooth=[9],n_dims=3):
    """
    Plot the PCA tuning spaces
    :param fname: Filename of the neo data
    :param unit_num: unit number to use
    :param p_smooth: [optional] If using derivative, this is where the smooth data live
    :param deriv_smooth: If using derivative, tells us what derivative smoothing to use
    :return:
    """

    # Get the standard data, from which the PCA will be computed
    blk = neoUtils.get_blk(fname)
    cbool = neoUtils.get_Cbool(blk)
    varlist = ['M','F','TH','PHIE']
    X = GLM.create_design_matrix(blk,varlist)
    sp = neoUtils.concatenate_sp(blk)['cell_{}'.format(unit_num)]
    r = neoUtils.get_rate_b(blk,unit_num)[0]

    # If smoothing directory is given, then add the derivative data
    if p_smooth is not None:
        blk_smooth = GLM.get_blk_smooth(fname,p_smooth)
        Xdot = GLM.get_deriv(blk,blk_smooth,varlist,deriv_smooth)[0]
        X = np.concatenate([X,Xdot],axis=1)
        X[np.isnan(X)]=0
    else:
        print('\tNot using derivative information')
    scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
    X_scale = np.zeros_like(X)
    X_scale[cbool,:] = scaler.fit_transform(X[cbool,:])
    pca = sklearn.decomposition.PCA()
    X_pcs = np.zeros_like(X)
    X_pcs[cbool,:] = pca.fit_transform(X_scale[cbool,:])
    for ii in range(n_dims):
        var = X_pcs[:,ii]
        response,edges = varTuning.stim_response_hist(var,sp,cbool)

    response,edges1,edges2 = varTuning.joint_response_hist(X_pcs[:,0],
                                                           X_pcs[:,1],
                                                           sp,
                                                           cbool,
                                                           40)

    response,edges1,edges2 = varTuning.joint_response_hist(X_pcs[:,-1],
                                                           X_pcs[:,-2],
                                                           sp,
                                                           cbool,
                                                           40)





def main(p_load,p_save):
    for f in glob.glob(os.path.join(p_load,'*.h5')):
        print('Working on {}'.format(os.path.basename(f)))
        blk = neoUtils.get_blk(f)
        num_units = len(blk.channel_indexes[-1].units)
        for unit_num in range(num_units):
            try:
                plot_joint_spaces(blk,unit_num,p_save=p_save)
            except:
                print('Issue with {}'.format(os.path.basename(f)))

if __name__=='__main__':
    p_load = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO')
    p_save = os.path.join(os.environ['BOX_PATH'], r'__VG3D\_deflection_trials\_NEO\results')
    main(p_load,p_save)
