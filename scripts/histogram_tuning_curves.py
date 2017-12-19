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

def mymz_space(blk,unit_num,save_tgl=False,p_save=None,im_ext='png',dpi_res=300):

    root = neoUtils.get_root(blk,unit_num)
    Cbool = neoUtils.get_Cbool(blk)
    M = neoUtils.get_var(blk).magnitude[Cbool,:]
    r,b = neoUtils.get_rate_b(blk,unit_num,sigma=5*pq.ms)
    r = r[Cbool]
    response, var1_edges,var2_edges = varTuning.joint_response_hist(M[:,1]*1e6,M[:,2]*1e6,r,bins = 50,min_obs=2)
    ax = varTuning.plot_joint_response(response,var1_edges,var2_edges,contour=True)

    ax.patch.set_color([0.2,0.2,0.2])
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
    Cbool = neoUtils.get_Cbool(blk)
    MB = mechanics.get_MB_MD(M)[0].magnitude.ravel()
    MB[np.invert(Cbool)]=0
    r, b = neoUtils.get_rate_b(blk, unit_num, sigma=2 * pq.ms)
    MB_bayes,edges = varTuning.stim_response_hist(MB*1e6,r,nbins=100,min_obs=5)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(edges[:-1],MB_bayes,'o',color='k')
    ax.set_ylabel('Spike Rate (sp/s)')
    ax.set_xlabel('Bending Moment ($\mu$N-m)')
    plt.tight_layout()
    if save_tgl:
        plt.savefig('./figs/{}_MB_tuning.{}'.format(root,im_ext),dpi=dpi_res)
        plt.close('all')


def phase_plots(blk,unit_num,save_tgl=False,p_save=None,im_ext='png',dpi_res=300):
    ''' Plot Phase planes for My and Mz'''
    root = neoUtils.get_root(blk, unit_num)
    Cbool = neoUtils.get_Cbool(blk)
    M = neoUtils.get_var(blk).magnitude
    r,b = neoUtils.get_rate_b(blk,unit_num,sigma=5*pq.ms)
    r = r[Cbool]
    Mdot = mechanics.get_deriv(M)[Cbool,:]
    M = M[Cbool,:]

    My_response,My_edges,Mydot_edges = varTuning.joint_response_hist(M[:, 1]*1e6, Mdot[:, 1]*1e6, r, 100,min_obs=5)
    axy = varTuning.plot_joint_response(My_response,My_edges,Mydot_edges,contour=True)

    Mz_response,Mz_edges,Mzdot_edges = varTuning.joint_response_hist(M[:, 2]*1e6, Mdot[:, 2]*1e6, r, 100,min_obs=5)
    axz = varTuning.plot_joint_response(Mz_response,Mz_edges,Mzdot_edges,contour=True)

    axy.set_title('M$_y$ Phase Plane')
    axz.set_title('M$_z$ Phase Plane')

    axy.set_xlabel('M$_y$ ($\mu$N-m)')
    axy.set_ylabel('M$_\dot{y}$ ($\mu$N-m/ms)')

    axz.set_xlabel('M$_z$ ($\mu$N-m)')
    axz.set_ylabel('M$_\dot{z}$ ($\mu$N-m/ms)')

    axy.grid('off')
    axy.set_facecolor([0.2, 0.2, 0.2])
    axz.grid('off')
    axz.set_facecolor([0.2, 0.2, 0.2])

    plt.sca(axy)
    axy.axis('normal')
    plt.tight_layout()
    if save_tgl:
        if p_save is None:
            raise ValueError("figure save location is required")
        else:
            plt.savefig(os.path.join(p_save,'{}_My_phaseplane.{}'.format(root,im_ext)),dpi=dpi_res)

    plt.sca(axz)
    axz.axis('normal')
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
    Cbool = neoUtils.get_Cbool(blk)

    Fx[np.invert(Cbool)] = 0
    r, b = neoUtils.get_rate_b(blk, unit_num, sigma=2 * pq.ms)
    Fx_bayes, edges = varTuning.stim_response_hist(Fx * 1e6, b, nbins=50, min_obs=5)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(edges[:-1], Fx_bayes*1000, 'o', color='k')
    ax.set_ylabel('Spike Rate (sp/s)')
    ax.set_xlabel('Axial Force ($\mu$N-m)')
    plt.tight_layout()
    if save_tgl:
        plt.savefig('./figs/{}_Fx_tuning.{}'.format(root,im_ext), dpi=dpi_res)
        plt.close('all')


def calc_all_mech_hists(p_load,p_save):
    all_F_edges = []
    all_M_edges = []
    all_F_bayes = []
    all_M_bayes = []
    all_MB_edges = []
    all_MD_edges = []
    all_MD_bayes = []
    all_MB_bayes = []
    ID = []

    for f in glob.glob(os.path.join(p_load,'rat*.h5')):
        print(os.path.basename(f))
        blk = neoUtils.get_blk(f)
        Cbool = neoUtils.get_Cbool(blk)
        for unit in blk.channel_indexes[-1].units:
            unit_num = int(unit.name[-1])
            r, b = neoUtils.get_rate_b(blk, unit_num, sigma=5 * pq.ms)
            root = neoUtils.get_root(blk,unit_num)
            M = neoUtils.get_var(blk).magnitude[Cbool,:]
            F = neoUtils.get_var(blk,'F').magnitude[Cbool,:]

            r = r[Cbool]


            MB, MD = neoUtils.get_MB_MD(M)
            M_bayes = np.empty([50,3])
            F_bayes = np.empty([50, 3])

            M_edges = np.empty([51, 3])
            F_edges = np.empty([51, 3])

            for ii in range(3):
                F_bayes[:, ii], F_edges[:, ii] = varTuning.stim_response_hist(F[:, ii] * 1e6, r, nbins=50, min_obs=5)
                M_bayes[:, ii], M_edges[:, ii] = varTuning.stim_response_hist(M[:, ii] * 1e6, r, nbins=50, min_obs=5)

            MB_bayes, MB_edges = varTuning.stim_response_hist(MB.squeeze() * 1e6, r, nbins=50, min_obs=5)
            MD_bayes, MD_edges,_,_ = varTuning.angular_response_hist(MD.squeeze(), r, nbins=50)

            plt.close('all')

            all_F_edges.append(F_edges)
            all_M_edges.append(M_edges)
            all_MB_edges.append(MB_edges)
            all_MD_edges.append(MD_edges)

            all_F_bayes.append(F_bayes)
            all_M_bayes.append(M_bayes)
            all_MB_bayes.append(MB_bayes)
            all_MD_bayes.append(MD_bayes)
            ID.append(root)

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


def calc_world_geom_hist(p_load,p_save):
    ID = []
    all_R_bayes = []
    all_TH_bayes = []
    all_PHIE_bayes = []
    all_ZETA_bayes = []

    all_R_edges = []
    all_TH_edges = []
    all_PHIE_edges = []
    all_ZETA_edges = []


    for f in glob.glob(os.path.join(p_load,'rat*.h5')):
        print(os.path.basename(f))
        blk = neoUtils.get_blk(f)
        Cbool = neoUtils.get_Cbool(blk)

        for unit in blk.channel_indexes[-1].units:
            unit_num = int(unit.name[-1])
            r, b = neoUtils.get_rate_b(blk, unit_num, sigma=5 * pq.ms)
            root = neoUtils.get_root(blk,unit_num)
            ID.append(root)

            R = neoUtils.get_var(blk,'Rcp').magnitude[Cbool]
            TH = neoUtils.get_var(blk,'TH').magnitude[Cbool]
            PHIE = neoUtils.get_var(blk, 'PHIE').magnitude[Cbool]
            ZETA = neoUtils.get_var(blk, 'ZETA').magnitude[Cbool]

            r = r[Cbool]

            R_bayes, R_edges = varTuning.stim_response_hist(R.ravel(), r, nbins=50, min_obs=5)
            TH_bayes, TH_edges = varTuning.stim_response_hist(TH.ravel(), r, nbins=50, min_obs=5)
            PHIE_bayes, PHIE_edges = varTuning.stim_response_hist(PHIE.ravel(), r, nbins=50,min_obs=5)
            ZETA_bayes, ZETA_edges = varTuning.stim_response_hist(ZETA.ravel(), r, nbins=50,min_obs=5)

            plt.close('all')
            all_R_bayes.append(R_bayes)
            all_TH_bayes.append(TH_bayes)
            all_PHIE_bayes.append(PHIE_bayes)
            all_ZETA_bayes.append(ZETA_bayes)


            all_R_edges.append(R_edges)
            all_TH_edges.append(TH_edges)
            all_PHIE_edges.append(PHIE_edges)
            all_ZETA_edges.append(ZETA_edges)


    np.savez(os.path.join(p_save, 'world_geom_hists.npz'),
             all_R_bayes=all_R_bayes,
             all_TH_bayes=all_TH_bayes,
             all_PHIE_bayes=all_PHIE_bayes,
             all_ZETA_bayes=all_ZETA_bayes,
             all_R_edges=all_R_edges,
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
    all_R_bayes = dat['all_R_bayes']
    all_TH_bayes = dat['all_TH_bayes']
    all_PHIE_bayes = dat['all_PHIE_bayes']
    all_ZETA_bayes = dat['all_ZETA_bayes']
    all_R_edges = dat['all_R_edges']
    all_TH_edges = dat['all_TH_edges']
    all_PHIE_edges = dat['all_PHIE_edges']
    all_ZETA_edges = dat['all_ZETA_edges']
    ID = dat['ID']


    for ii in range(len(ID)):
        fig = plt.figure()

        ax = fig.add_subplot(2,2,1)
        plt.plot(all_R_edges[ii][:-1], all_R_bayes[ii], 'ko')
        ax.set_title('R')

        ax = fig.add_subplot(2,2,2)
        plt.plot(all_TH_edges[ii][:-1], all_TH_bayes[ii], 'ko')
        # ax.set_xlim([-np.pi/2,np.pi/2])
        ax.set_title('TH')

        ax = fig.add_subplot(2,2,3)
        plt.plot(all_PHIE_edges[ii][:-1], all_PHIE_bayes[ii], 'ko')
        # ax.set_xlim([-np.pi/2, np.pi/2])
        ax.set_title('PHIE')

        ax = fig.add_subplot(2,2,4)
        plt.plot(all_ZETA_edges[ii][:-1], all_ZETA_bayes[ii], 'ko')
        ax.set_title('ZETA')


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
            mymz_space(blk, unit_num, p_save=p_save, save_tgl=True, im_ext='png', dpi_res=300)
            phase_plots(blk, unit_num, p_save=p_save, save_tgl=True, im_ext='png', dpi_res=300)

if __name__=='__main__':
    p_load = r'C:\Users\guru\Box Sync\__VG3D\_deflection_trials\_NEO'
    p_save = r'C:\Users\guru\Box Sync\__VG3D\_deflection_trials\_NEO\results'
    calc_all_mech_hists(p_load,p_save)
    calc_world_geom_hist(p_load,p_save)
    plot_all_geo_hists(save_tgl=True)
    plot_all_mech_hists(save_tgl=True)
    plot_joint_spaces(p_load,p_save)