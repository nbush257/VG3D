import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np
import quantities as pq
import varTuning
import mechanics
import neoUtils
sns.set()

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['font.sans-serif'] = 'Arial'
dpi_res = 300
def nl(x,k_scale=80):
    k = 1/np.percentile(x,k_scale)
    z = 2/(1+np.exp(-k*x)) - 1
    invnl = lambda y: -1/k*(np.log(1-y)-np.log(1+y))
    return z,invnl



def get_bins(x,nbins):
    bin_counts = len(x)/nbins


def mymz_space(blk,unit_num,bin_stretch=False,save_tgl=False,p_save=None,im_ext='png',dpi_res=300):

    root = neoUtils.get_root(blk,unit_num)
    use_flags = neoUtils.get_Cbool(blk)
    M = neoUtils.get_var(blk).magnitude
    sp = neoUtils.concatenate_sp(blk)['cell_{}'.format(unit_num)]
    idx = np.all(np.isfinite(M),axis=1)
    if bin_stretch:
        MY = np.empty(M.shape[0])
        MZ = np.empty(M.shape[0])
        MY[idx], logit_y = nl(M[idx, 1],90)
        MZ[idx], logit_z = nl(M[idx, 2],90)
    else:
        MY = M[:,1]*1e-6
        MZ = M[:,2]*1e-6


    response, var1_edges,var2_edges = varTuning.joint_response_hist(MY,MZ,sp,use_flags,bins = 100,min_obs=15)
    if bin_stretch:
        var1_edges = logit_y(var1_edges)
        var2_edges = logit_z(var2_edges)
    else:
        pass
    ax = varTuning.plot_joint_response(response,var1_edges,var2_edges,contour=False)
    ax.axvline(color='k',linewidth=1)
    ax.axhline(color='k',linewidth=1)
    ax.patch.set_color([0.6,0.6,0.6])

    mask = response.mask.__invert__()
    if not mask.all():
        ax.set_ylim(var2_edges[np.where(mask)[0].min()], var2_edges[np.where(mask)[0].max()])
        ax.set_xlim(var1_edges[np.where(mask)[1].min()], var1_edges[np.where(mask)[1].max()])

    ax.set_xlabel('M$_y$ ($\mu$N-m)')
    ax.set_ylabel('M$_z$ ($\mu$N-m)')
    plt.draw()
    plt.tight_layout()
    if save_tgl:
        if p_save is None:
            raise ValueError("figure save location is required")
        else:
            plt.savefig(os.path.join(p_save,'{}_mymz.{}'.format(root,im_ext)),dpi=dpi_res)
            plt.close('all')


def MB_curve(blk,unit_num,save_tgl=False,im_ext='svg',dpi_res=300):
    root = neoUtils.get_root(blk, unit_num)
    M = neoUtils.get_var(blk)
    use_flags = neoUtils.get_Cbool(blk)
    MB = mechanics.get_MB_MD(M)[0].magnitude.ravel()
    MB[np.invert(use_flags)]=0
    sp = neoUtils.concatenate_sp(blk)['cell_{}'.format(unit_num)]
    r, b = neoUtils.get_rate_b(blk, unit_num, sigma=5 * pq.ms)

    MB_bayes,edges = varTuning.stim_response_hist(MB*1e6,r,use_flags,nbins=100,min_obs=5)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(edges[:-1],MB_bayes,'o',color='k')
    ax.set_ylabel('Spike Rate (sp/s)')
    ax.set_xlabel('Bending Moment ($\mu$N-m)')
    plt.tight_layout()
    if save_tgl:
        plt.savefig('./figs/{}_MB_tuning.{}'.format(root,im_ext),dpi=dpi_res)
        plt.close('all')


def phase_plots(blk,unit_num,save_tgl=False,bin_stretch=False,p_save=None,im_ext='png',dpi_res=300):
    ''' Plot Phase planes for My and Mz'''
    root = neoUtils.get_root(blk, unit_num)
    M = neoUtils.get_var(blk).magnitude
    sp = neoUtils.concatenate_sp(blk)['cell_{}'.format(unit_num)]
    r, b = neoUtils.get_rate_b(blk, unit_num, sigma=5 * pq.ms)


    use_flags = neoUtils.get_Cbool(blk)
    Mdot = mechanics.get_deriv(M)


    if bin_stretch:
        raise Exception('Not finished with use_flags')
        # MY, logit_y = nl(M[idx, 1], 90)
        # MZ, logit_z = nl(M[idx, 2], 90)
        # MY_dot, logit_ydot = nl(Mdot[idx, 1], 95)
        # MZ_dot, logit_zdot = nl(Mdot[idx, 2], 95)

    else:
        MY = M[:, 1] * 1e-6
        MZ = M[:, 2] * 1e-6
        MY_dot = Mdot[:, 1] * 1e-6
        MZ_dot = Mdot[:, 2] * 1e-6

    My_response,My_edges,Mydot_edges = varTuning.joint_response_hist(MY, MY_dot, r, use_flags, [100,30],min_obs=15)
    Mz_response,Mz_edges,Mzdot_edges = varTuning.joint_response_hist(MZ, MZ_dot, r, use_flags, [100,30],min_obs=15)


    if bin_stretch:
        My_edges = logit_y(My_edges)
        Mz_edges = logit_z(Mz_edges)
        Mydot_edges = logit_ydot(Mydot_edges)
        Mzdot_edges = logit_zdot(Mzdot_edges)
    else:
        pass

    axy = varTuning.plot_joint_response(My_response,My_edges,Mydot_edges,contour=False)
    axz = varTuning.plot_joint_response(Mz_response,Mz_edges,Mzdot_edges,contour=False)

    # Set bounds
    y_mask = My_response.mask.__invert__()
    if not y_mask.all():
        axy.set_ylim(Mydot_edges[np.where(y_mask)[0].min()], Mydot_edges[np.where(y_mask)[0].max()])
        axy.set_xlim(My_edges[np.where(y_mask)[1].min()], My_edges[np.where(y_mask)[1].max()])

    z_mask = Mz_response.mask.__invert__()
    if not z_mask.all():
        axz.set_ylim(Mzdot_edges[np.where(z_mask)[0].min()], Mzdot_edges[np.where(z_mask)[0].max()])
        axz.set_xlim(Mz_edges[np.where(z_mask)[1].min()], Mz_edges[np.where(z_mask)[1].max()])

    # other annotations
    axy.set_title('M$_y$ Phase Plane')
    axz.set_title('M$_z$ Phase Plane')

    axy.set_xlabel('M$_y$ ($\mu$N-m)')
    axy.set_ylabel('M$_\dot{y}$ ($\mu$N-m/ms)')

    axz.set_xlabel('M$_z$ ($\mu$N-m)')
    axz.set_ylabel('M$_\dot{z}$ ($\mu$N-m/ms)')

    axy.grid('off')
    axy.set_facecolor([0.6, 0.6, 0.6])
    axy.axvline(color='k',linewidth=1)
    axy.axhline(color='k',linewidth=1)

    axz.grid('off')
    axz.set_facecolor([0.6, 0.6, 0.6])
    axz.axvline(color='k', linewidth=1)
    axz.axhline(color='k', linewidth=1)


    plt.sca(axy)
    plt.tight_layout()
    if save_tgl:
        if p_save is None:
            raise ValueError("figure save location is required")
        else:
            plt.savefig(os.path.join(p_save,'{}_My_phaseplane.{}'.format(root,im_ext)),dpi=dpi_res)

    plt.sca(axz)
    plt.tight_layout()
    if save_tgl:
        if p_save is None:
            raise ValueError("figure save location is required")
        else:
            plt.savefig(os.path.join(p_save,'{}_Mz_phaseplane.{}'.format(root,im_ext)),dpi=dpi_res)
        plt.close('all')


def FX_plots(blk,unit_num,save_tgl=False,im_ext='svg',dpi_res=300):
    root = neoUtils.get_root(blk, unit_num)
    F = neoUtils.get_var(blk,'F')
    Fx = F.magnitude[:,0]
    use_flags = neoUtils.get_Cbool(blk)
    sp = neoUtils.concatenate_sp(blk)['cell_{}'.format(unit_num)]
    r, b = neoUtils.get_rate_b(blk, unit_num, sigma=5 * pq.ms)

    Fx[np.invert(use_flags)] = 0

    Fx_bayes, edges = varTuning.stim_response_hist(Fx * 1e6, r, use_flags, nbins=50, min_obs=5)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(edges[:-1], Fx_bayes*1000, 'o', color='k')
    ax.set_ylabel('Spike Rate (sp/s)')
    ax.set_xlabel('Axial Force ($\mu$N-m)')
    plt.tight_layout()
    if save_tgl:
        plt.savefig('./figs/{}_Fx_tuning.{}'.format(root,im_ext), dpi=dpi_res)
        plt.close('all')


def calc_all_mech_hists(p_load,p_save,n_bins=100):
    """
    Since calculation takes so long on getting the histograms (mostly loading of data)
    we want to calculate them once and save the data.

    This calculates the mechanics.

    :param p_load: Location where all the neo h5 files live
    :param p_save: Location to save the output data files
    :param n_bins: Number of bins in with which to split the data
    :return None: Saves a 'mech_histograms.npz' file.
    """

    # TODO: This is currently pretty gross, it is really too hardcoded (I wrote it in a car). Do better.
    # TODO: Combine with geometry

    # Case in point:
    all_F_edges = []
    all_M_edges = []
    all_F_bayes = []
    all_M_bayes = []
    all_MB_edges = []
    all_MD_edges = []
    all_MD_bayes = []
    all_MB_bayes = []
    ID = []

    # Loop all neo files
    for f in glob.glob(os.path.join(p_load,'rat*.h5')):
        print(os.path.basename(f))
        blk = neoUtils.get_blk(f)
        Cbool = neoUtils.get_Cbool(blk)
        # Loop all units
        for unit in blk.channel_indexes[-1].units:
            unit_num = int(unit.name[-1])

            # grab needed variables
            r, b = neoUtils.get_rate_b(blk, unit_num, sigma=5 * pq.ms)
            sp = neoUtils.concatenate_sp(blk)['cell_{}'.format(unit_num)]
            root = neoUtils.get_root(blk,unit_num)
            M = neoUtils.get_var(blk).magnitude
            F = neoUtils.get_var(blk,'F').magnitude
            MB, MD = neoUtils.get_MB_MD(M)

            # init histograms
            M_bayes = np.empty([n_bins,3])
            F_bayes = np.empty([n_bins, 3])

            M_edges = np.empty([n_bins+1, 3])
            F_edges = np.empty([n_bins+1, 3])

            #calculate tuning curves (seperately on each dimension)
            for ii in range(3):
                F_bayes[:, ii], F_edges[:, ii] = varTuning.stim_response_hist(F[:, ii] * 1e6, r, Cbool, nbins=n_bins, min_obs=5)
                M_bayes[:, ii], M_edges[:, ii] = varTuning.stim_response_hist(M[:, ii] * 1e6, r, Cbool, nbins=n_bins, min_obs=5)
            MB_bayes, MB_edges = varTuning.stim_response_hist(MB.squeeze() * 1e6, r, Cbool, nbins=n_bins, min_obs=5)
            MD_bayes, MD_edges,_,_ = varTuning.angular_response_hist(MD.squeeze(), r, Cbool, nbins=n_bins)
            plt.close('all')

            # append to output lists
            all_F_edges.append(F_edges)
            all_M_edges.append(M_edges)
            all_MB_edges.append(MB_edges)
            all_MD_edges.append(MD_edges)

            all_F_bayes.append(F_bayes)
            all_M_bayes.append(M_bayes)
            all_MB_bayes.append(MB_bayes)
            all_MD_bayes.append(MD_bayes)
            ID.append(root)
    # save
    np.savez(os.path.join(p_save,'mech_histograms.npz'),
             all_F_bayes=all_F_bayes,
             all_F_edges=all_F_edges,
             all_M_bayes=all_M_bayes,
             all_M_edges=all_M_edges,
             all_MB_bayes=all_MB_bayes,
             all_MB_edges=all_MB_edges,
             all_MD_bayes=all_MD_bayes,
             all_MD_edges=all_MD_edges,
             ID=ID
             )


def calc_world_geom_hist(p_load,p_save,n_bins=100):
    """
     Since calculation takes so long on getting the histograms (mostly loading of data)
    we want to calculate them once and save the data.

    This calculates the Geometry.

    :param p_load: Location where all the neo h5 files live
    :param p_save: Location to save the output data files
    :param n_bins: Number of bins in with which to split the data
    :return None: Saves a 'world_geom_hists.npz' file.
    """
    # init
    ID = []
    all_S_bayes = []
    all_TH_bayes = []
    all_PHIE_bayes = []
    all_ZETA_bayes = []

    all_S_edges = []
    all_TH_edges = []
    all_PHIE_edges = []
    all_ZETA_edges = []

    # loop files
    for f in glob.glob(os.path.join(p_load,'rat*.h5')):
        # load in
        print(os.path.basename(f))
        blk = neoUtils.get_blk(f)

        # get contact
        Cbool = neoUtils.get_Cbool(blk)
        use_flags = neoUtils.concatenate_epochs(blk)

        # get vars
        S = neoUtils.get_var(blk, 'S').magnitude

        TH = neoUtils.get_var(blk, 'TH').magnitude
        neoUtils.center_var(TH, use_flags)

        PHIE = neoUtils.get_var(blk, 'PHIE').magnitude
        neoUtils.center_var(PHIE, use_flags)

        ZETA = neoUtils.get_var(blk, 'ZETA').magnitude
        neoUtils.center_var(ZETA, use_flags)

        # loop units
        for unit in blk.channel_indexes[-1].units:
            # get unit info
            unit_num = int(unit.name[-1])
            r, b = neoUtils.get_rate_b(blk, unit_num, sigma=5 * pq.ms)
            sp = neoUtils.concatenate_sp(blk)['cell_{}'.format(unit_num)]
            root = neoUtils.get_root(blk,unit_num)
            ID.append(root)

            # Create hists
            S_bayes, S_edges = varTuning.stim_response_hist(S.ravel(), r, Cbool, nbins=n_bins, min_obs=5)
            TH_bayes, TH_edges = varTuning.stim_response_hist(TH.ravel(), r, Cbool, nbins=n_bins, min_obs=5)
            PHIE_bayes, PHIE_edges = varTuning.stim_response_hist(PHIE.ravel(), r, Cbool, nbins=n_bins,min_obs=5)
            ZETA_bayes, ZETA_edges = varTuning.stim_response_hist(ZETA.ravel(), r, Cbool, nbins=n_bins,min_obs=5)

            # append outputs
            plt.close('all')
            all_S_bayes.append(S_bayes)
            all_TH_bayes.append(TH_bayes)
            all_PHIE_bayes.append(PHIE_bayes)
            all_ZETA_bayes.append(ZETA_bayes)

            all_S_edges.append(S_edges)
            all_TH_edges.append(TH_edges)
            all_PHIE_edges.append(PHIE_edges)
            all_ZETA_edges.append(ZETA_edges)


    np.savez(os.path.join(p_save, 'world_geom_hists.npz'),
             all_S_bayes=all_S_bayes,
             all_TH_bayes=all_TH_bayes,
             all_PHIE_bayes=all_PHIE_bayes,
             all_ZETA_bayes=all_ZETA_bayes,
             all_S_edges=all_S_edges,
             all_TH_edges=all_TH_edges,
             all_PHIE_edges=all_PHIE_edges,
             all_ZETA_edges=all_ZETA_edges,
             ID=ID
             )


def plot_all_mech_hists(f=None, im_ext='png', save_tgl=False):
    if f is None:
        f = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results\mech_histograms.npz')

    p_save = os.path.split(f)[0]
    dat = np.load(f)
    all_F_edges = dat['all_F_edges']
    all_M_edges = dat['all_M_edges']
    all_F_bayes = dat['all_F_bayes']
    all_M_bayes = dat['all_M_bayes']
    all_MB_edges = dat['all_MB_edges']
    all_MD_edges = dat['all_MD_edges']
    all_MD_bayes = dat['all_MD_bayes']
    all_MB_bayes = dat['all_MB_bayes']
    ID = dat['ID']


    for ii in range(len(ID)):
        fig = plt.figure()

        ax = fig.add_subplot(2,4,1)
        plt.plot(all_M_edges[ii][:-1, 0], all_M_bayes[ii][:, 0], 'ko')
        ax.set_title('M_x')

        ax = fig.add_subplot(2,4,2)
        plt.plot(all_M_edges[ii][:-1, 1], all_M_bayes[ii][:, 1], 'ko')
        ax.set_title('M_y')

        ax = fig.add_subplot(2,4,3)
        plt.plot(all_M_edges[ii][:-1, 2], all_M_bayes[ii][:, 2], 'ko')
        ax.set_title('M_z')

        ax = fig.add_subplot(2,4,5)
        plt.plot(all_F_edges[ii][:-1, 0], all_F_bayes[ii][:, 0], 'ko')
        ax.set_title('F_x')

        ax= fig.add_subplot(2,4,6)
        plt.plot(all_F_edges[ii][:-1, 1], all_F_bayes[ii][:, 1], 'ko')
        ax.set_title('F_y')

        ax = fig.add_subplot(2,4,7)
        plt.plot(all_F_edges[ii][:-1, 2], all_F_bayes[ii][:, 2], 'ko')
        ax.set_title('F_z')

        ax = fig.add_subplot(244)
        plt.plot(all_MB_edges[ii][:-1], all_MB_bayes[ii], 'ko')
        ax.set_title('MB_tuning')
        ax = fig.add_subplot(248,projection='polar')
        theta,L_dir = varTuning.get_PD_from_hist(all_MD_edges[ii][:-1],all_MD_bayes[ii])
        plt.plot(all_MD_edges[ii][:-1], all_MD_bayes[ii], 'ko')
        ax.annotate('',
                    xy=(theta, L_dir*np.nanmax(all_MD_bayes[ii])),
                    xytext=(0, 0),
                    arrowprops={'arrowstyle': 'simple,head_width=1','linewidth': 1, 'facecolor':'g','alpha': 0.3}
                    )
        ax.set_title('MD_tuning')

        fig.suptitle('{}'.format(ID[ii]))
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.pause(0.2)
        if save_tgl:
            plt.savefig(os.path.join(p_save, '{}_stim_var_response.{}'.format(ID[ii], im_ext)), dpi=300)
            plt.close()


def plot_all_geo_hists(f=None, im_ext='png',save_tgl=True):

    if f is None:
        f = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results\world_geom_hists.npz')

    p_save = os.path.split(f)[0]
    dat = np.load(f)
    all_S_bayes = dat['all_S_bayes']
    all_TH_bayes = dat['all_TH_bayes']
    all_PHIE_bayes = dat['all_PHIE_bayes']
    all_ZETA_bayes = dat['all_ZETA_bayes']
    all_S_edges = dat['all_S_edges']
    all_TH_edges = dat['all_TH_edges']
    all_PHIE_edges = dat['all_PHIE_edges']
    all_ZETA_edges = dat['all_ZETA_edges']
    ID = dat['ID']


    for ii in range(len(ID)):
        fig = plt.figure()

        ax = fig.add_subplot(2,2,1)
        plt.plot(all_S_edges[ii][:-1], all_S_bayes[ii], 'ko')
        ax.set_title('Arclength (m)')

        ax = fig.add_subplot(2,2,2)
        plt.plot(all_TH_edges[ii][:-1], all_TH_bayes[ii], 'ko')
        # ax.set_xlim([-np.pi/2,np.pi/2])
        ax.set_title(r'$\Delta\theta$')

        ax = fig.add_subplot(2,2,3)
        plt.plot(all_PHIE_edges[ii][:-1], all_PHIE_bayes[ii], 'ko')
        # ax.set_xlim([-np.pi/2, np.pi/2])
        ax.set_title(r'$\Delta\phi$')

        ax = fig.add_subplot(2,2,4)
        plt.plot(all_ZETA_edges[ii][:-1], all_ZETA_bayes[ii], 'ko')
        ax.set_title(r'$\Delta\zeta$')


        fig.suptitle('{}'.format(ID[ii]))
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.tight_layout()
        plt.pause(0.2)
        if save_tgl:
            plt.savefig(os.path.join(p_save, '{}_geo_var_response.{}'.format(ID[ii], im_ext)), dpi=300)
            plt.close()


def plot_joint_spaces(p_load,p_save):
    for f in glob.glob(os.path.join(p_load,'rat*.h5')):
        print(os.path.basename(f))
        blk = neoUtils.get_blk(f)
        for unit in blk.channel_indexes[-1].units:
            unit_num = int(unit.name[-1])
            try:
                mymz_space(blk, unit_num, p_save=p_save, save_tgl=True, im_ext='png', dpi_res=300)
                phase_plots(blk, unit_num, p_save=p_save, save_tgl=True, im_ext='png', dpi_res=300)
            except:
                print('File {} did not create jointplots'.format(neoUtils.get_root(blk,unit_num)))
                pass

if __name__=='__main__':
    p_load = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO')
    p_save = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')
    calc_all_mech_hists(p_load,p_save)
    calc_world_geom_hist(p_load,p_save)
    plot_all_geo_hists(save_tgl=True)
    plot_all_mech_hists(save_tgl=True)
    plot_joint_spaces(p_load,p_save)