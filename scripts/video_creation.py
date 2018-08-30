import pims
import shutil
import math
import pyaudio
from scipy.io.matlab import loadmat
from scipy.io import wavfile
import scipy
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import os
import glob
import numpy as np
import h5py
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import pywt
from optparse import OptionParser
sns.set_style('ticks')

#

def get_whisker_2d_pts(h5_file,frame_no):
    ''' we are skipping getting the 2D points because the files are too unweildy'''
    pass

def get_whisker_3D_pts(mat_file,frame_no):
    xx = mat_file['xw3dm'][0][frame_no].ravel()
    yy = mat_file['yw3dm'][0][frame_no].ravel()
    zz = mat_file['zw3dm'][0][frame_no].ravel()
    return(xx,yy,zz)

def get_spike_cp_coords(dat3D,cell_num=0):
    sp = dat3D['sp'][0][cell_num]
    framesamps = dat3D['framesamps']
    x = dat3D['Xcp']

    x_spike = np.empty_like(x);x_spike[:]=np.nan
    y_spike = np.empty_like(x);y_spike[:]=np.nan
    z_spike = np.empty_like(x);z_spike[:]=np.nan

    x_spike[sp] = x[sp]
    y_spike[sp] = y[sp]
    z_spike[sp] = z[sp]

def spikes_in_frame(dat3D,frame,cell_num=0):
    framesamps = dat3D['framesamps'][frame-1:frame+1,0]
    sp = dat3D['sp'][0][cell_num]
    spikes = np.where(np.logical_and(sp>framesamps[0],sp<framesamps[1]))[0]
    return len(spikes)

def get_ms_idx(dat3D,frame,t_range=500):
    #range is in ms
    start_time = dat3D['frametimes'][frame][0]-t_range/1000.
    stop_time = dat3D['frametimes'][frame][0]+t_range/1000.
    ms_idx = np.array([np.round(start_time*1000.),np.round(stop_time*1000.)],dtype='int')
    return ms_idx

def get_bounds(dat3D,frames):
    maxx=[]
    maxy=[]
    maxz=[]
    minx=[]
    miny=[]
    minz=[]
    bds = {}
    for frame in frames:
        xx, yy, zz = get_whisker_3D_pts(dat3D, frame)
        maxx.append(np.nanmax(xx-xx[0]))
        maxy.append(np.nanmax(yy-yy[0]))
        maxz.append(np.nanmax(zz-zz[0]))

        minx.append(np.nanmin(xx-xx[0]))
        miny.append(np.nanmin(yy-yy[0]))
        minz.append(np.nanmin(zz-zz[0]))
    bds['maxx'] = np.nanmax(maxx)
    bds['maxy'] = np.nanmax(maxy)
    bds['maxz'] = np.nanmax(maxz)

    bds['minx'] = np.nanmin(minx)
    bds['miny'] = np.nanmin(miny)
    bds['minz'] = np.nanmin(minz)
    return bds

def get_spbool(dat_1K,cell_num=0):
    sp_1k = dat_1K['sp'][0][CELL_NUM]
    l = dat_1K['filtvars'][0]['M'][0].shape[0]
    spbool = np.zeros([l,1],dtype='int')
    spbool[sp_1k]=1
    return spbool

def gen_spike_track(dat_3D,frames,fps=50):
    # fps is the desired framerate of the output video
    output = np.array([],dtype = 'float32')
    w_length=100
    bitrate = 44100
    length = 1./fps# length of a frame in seconds
    num_samps_per_frame = int(bitrate*length)
    spikeshape = pywt.ContinuousWavelet('mexh').wavefun(length=w_length)[0]
    for ii,frame in enumerate(frames):
        frame_sound = np.zeros(num_samps_per_frame,dtype='float32')
        num_spikes = spikes_in_frame(dat_3D, frame)
        if  num_spikes> 0:
            frame_sound[:w_length*num_spikes]=np.tile(spikeshape,num_spikes).ravel()
        output = np.concatenate([output,frame_sound])
    return output

def get_rotation(dat_1K):
    """
    Extracts and Centers the rotation variables per contact
    :param dat_1K: a matfile containing the 1K sampled mechanics data
    :return: R, a [Tx2] matrix of deltaTheta, deltaPhi values over time (T)
    """
    TH = dat_1K['filtvars'][0]['TH'][0]
    PHI = dat_1K['filtvars'][0]['PHIE'][0]
    Cbool = dat_1K['use_flags'].astype('int')
    Cbool[np.isnan(Cbool)]=0
    starts = np.where(np.diff(Cbool.ravel())==1)[0]
    stops = np.where(np.diff(Cbool.ravel())==-1)[0]
    deltaTH = TH.copy()
    deltaPHI = PHI.copy()
    for start,stop in zip(starts,stops):
        deltaTH[start:stop+1,:] = deltaTH[start:stop+1] - deltaTH[start]
        deltaPHI[start:stop+1,:]= deltaPHI[start:stop+1] - deltaPHI[start]
    deltaTH[Cbool==0]=0
    deltaPHI[Cbool==0]=0
    return(np.concatenate([deltaTH,deltaPHI],axis=1))

def getRX(theta):
    """
    create a 3x3 rotation matrix around the x axis
    :param theta: theta  angle to rotate in radians
    :return:  3x3 rotation matrix
    """
    c =np.cos(theta)[0]
    s = np.sin(theta)[0]
    RX = np.array([[1.,0.,0.],
                   [0.,c,-s],
                   [0.,s,c]])
    return(RX)

def getRY(phi):
    """
    create a 3x3 rotation matrix around the Y axis
    :param phi: theta  angle to rotate in radians
    :return:  3x3 rotation matrix
    """
    c =np.cos(phi)[0]
    s = np.sin(phi)[0]
    RY = np.array([[c,0.,s],
                   [0.,1,0.],
                   [-s,0.,c]])

    return(RY)
def getRZ(zeta):
    """
    create a 3x3 rotation matrix around the z axis
    :param zeta: theta  angle to rotate in radians
    :return:  3x3 rotation matrix
    """
    c =np.cos(zeta)[0]
    s = np.sin(zeta)[0]
    RZ = np.array([[c,-s,0.],
                   [s,c,.0],
                   [0.,0.,1.]])
    return(RZ)
def rotate_pts(X,Y,Z,frame,dat_1K):
    """
    Takes arrays of X,Y,and Z points in frame [frame], and rotates all points according to the
    most recent reference frame
    :param X:
    :param Y:
    :param Z:
    :param frame:
    :param dat_1K:
    :return:
    """
    TH = dat_1K['filtvars'][0]['TH'][0]
    PHI = dat_1K['filtvars'][0]['PHIE'][0]
    ZETA = dat_1K['filtvars'][0]['ZETA'][0]
    Cbool = dat_1K['use_flags'].astype('int')
    Cbool[np.isnan(Cbool)]=0
    starts = np.where(np.diff(Cbool.ravel())==1)[0]

    REF = starts[starts<frame][-1]

    th = np.deg2rad(TH[REF])
    phi = np.deg2rad(PHI[REF])
    zeta = ZETA[REF]
    ROT = np.dot(np.dot(getRX(-zeta),getRY(-phi)),getRZ(-th))
    pts = np.vstack((X,Y,Z)).T
    pts = np.dot(pts,ROT)

    return(pts[:,0],pts[:,1],pts[:,2])

    # ================================================================ #
def main():
    print('Loading Data...')
    f_vid_front_load = glob.glob(os.path.join(p_vid_load,'rat{}*VG_{}_{}_Front.mkv'.format(RATNUM,WHISKER_ID,TRIAL)))[0]
    f_vid_top_load = glob.glob(os.path.join(p_vid_load,'rat{}*VG_{}_{}_Top.mkv'.format(RATNUM,WHISKER_ID,TRIAL)))[0]
    f_3D_data = glob.glob(os.path.join(p_3D_data, 'rat{}*VG_{}_{}_E3D_PROC.mat'.format(RATNUM, WHISKER_ID, TRIAL)))[0]
    f_2D_data = glob.glob(os.path.join(p_2D_data, 'rat{}*VG_{}_{}_toMerge.mat'.format(RATNUM, WHISKER_ID, TRIAL)))[0]
    f_1K_data = glob.glob(os.path.join(p_3D_data, 'rat{}*VG_{}_{}_1K.mat'.format(RATNUM, WHISKER_ID, TRIAL)))[0]

    if not os.path.isfile(f_vid_front_load):
        raise ValueError('Front Video is not a valid video file')
    elif not os.path.isfile(f_vid_top_load):
        raise ValueError('Top Video is not a valid video file')
    elif not os.path.isfile(f_2D_data):
        raise ValueError('2D Whisker Data is not a valid file')
    elif not os.path.isfile(f_3D_data):
        raise ValueError('3D Whisker Data is not a valid file')
    elif not os.path.isfile(f_1K_data):
        raise ValueError('1K Mechanics data is not a valid file')

    frames = np.arange(START_FRAME,STOP_FRAME,FRAME_STEP)

    # === LOAD IN DATA FILES === #
    v_front = pims.open(f_vid_front_load)
    v_top = pims.open(f_vid_top_load)
    # dat2D = h5py.File(f_2D_data,'r') # NOT IMPLEMENTING 2D TRACKING. Could possibly dp back projections?
    dat_3D = loadmat(f_3D_data)
    dat_1K = loadmat(f_1K_data)


    # === CALCULATE REQUIRED VARIABLES === #
    print('Calculating Vars...')
     # get spikes
    sp_1k = dat_1K['sp'][0][CELL_NUM]
    spbool = get_spbool(dat_1K,cell_num=CELL_NUM)

    # get a contact boolean
    Cbool = dat_1K['use_flags'].ravel()
    Cbool[np.isnan(Cbool)]=0
    Cbool = Cbool.astype('bool')

    # get moment and its bounds
    M = dat_1K['filtvars'][0]['M'][0]*10**6
    temp1 = get_ms_idx(dat_3D,frames[0],T_RANGE)[0]-T_RANGE
    temp2 = get_ms_idx(dat_3D,frames[-1],T_RANGE)[0]+T_RANGE
    maxM = np.nanmax(np.nanmax(M[temp1:temp2,:],axis=0))
    minM = np.nanmin(np.nanmin(M[temp1:temp2,:],axis=0))
    # set noncontact moment to zero
    M[np.invert(Cbool),:]=0

    F = dat_1K['filtvars'][0]['F'][0]*10**6
    maxF = np.nanmax(np.nanmax(F[temp1:temp2,:],axis=0))
    minF = np.nanmin(np.nanmin(F[temp1:temp2,:],axis=0))
    # set noncontact moment to zero
    F[np.invert(Cbool),:]=0
    # init spike frames for spike induced 3DCP plot
    spike_frames=[]

    R = get_rotation(dat_1K)
    maxR = np.nanmax(np.nanmax(R[temp1:temp2,:],axis=0))
    minR = np.nanmin(np.nanmin(R[temp1:temp2,:],axis=0))

    # set noncontact vars to zero
    M[np.invert(Cbool),:]=0
    F[np.invert(Cbool),:]=0
    R[np.invert(Cbool),:]=0

    # get 3D contact point
    Xcp = dat_3D['CPm'][:, 0]
    Ycp = dat_3D['CPm'][:, 1]
    Zcp = dat_3D['CPm'][:, 2]

    # get axis bounds on the 3D whisker
    bds = get_bounds(dat_3D, frames)

    # make sure we dont try to grab frames past the length of the video
    if (frames[-1]>=len(v_front)) or (frames[-1]>=len(v_top)):
        raise ValueError('Desired frame range exceeds size of videos')

    # === SET UP PLOTTING === #
    # Initialize subplot axes and positioning
    fig = plt.figure(figsize=(11,9),dpi=150)
    axVid = plt.subplot2grid((6,2),(0,0),rowspan=3,colspan=2)
    ax3 = plt.subplot2grid((6, 2), (3, 0), rowspan=3,projection='3d')
    axMoment = plt.subplot2grid((6, 2), (3, 1))
    axForce = plt.subplot2grid((6, 2), (4, 1))
    axRotation = plt.subplot2grid((6, 2), (5, 1))
    plt.tight_layout()

    # Set count for iteration of frame numbers
    count=0

    # === Loop over the frames to plot === #
    for frame in frames:
        print('\tFrame {} of {}'.format(frame,frames[-1]))

        # append the current contact point if a spike happened since the last frame
        if spikes_in_frame(dat_3D, frame)>0:
            spike_frames.append(frame)

        # ============ 2D video ============== #
        # get image
        If = v_front.get_frame(frame)[:,:,0]
        It = v_top.get_frame(frame)[:,:,0]
        I = np.concatenate([If,It],axis=1)

        # plot and format axes of 2D vid
        axVid.cla()
        axVid.imshow(I,cmap='gray')
        axVid.set_xticklabels([])
        axVid.set_yticklabels([])
        axVid.grid('off')
        axVid.set_title('2D video in Front and Top')
        axVid.set_xticks([])
        axVid.set_yticks([])

        # ============= plot 3D whisker and contact point ========================

        # grab 3D points for whisker and contact point history
        xx,yy,zz = get_whisker_3D_pts(dat_3D, frame)
        # xx,yy,zz = rotate_pts(xx,yy,zz,frame,dat_1K)

        xcp = Xcp[frame-HISTORY:frame]
        ycp = Ycp[frame-HISTORY:frame]
        zcp = Zcp[frame-HISTORY:frame]

        # xcp,ycp,zcp = rotate_pts(xcp,ycp,zcp,frame,dat_1K)
        # plot and format 3D whisker shape plotting.
        #       Multiplying by 1000 to convert meters to mm
        ax3.cla()
        ax3.axis('equal')
        # 3D whisker
        ax3.plot((xx-xx[0])*1000,(yy-yy[0])*1000,(zz-zz[0])*1000,'k')
        # Basepoint
        ax3.scatter(0, 0,0, 'o', c='r')
        # Contact point history
        ax3.scatter((xcp-xx[0])*1000,(ycp-yy[0])*1000,(zcp-zz[0])*1000,'ko',c=np.linspace(0,0.3,HISTORY))
        # Contact point when a spike occured
        ax3.plot((Xcp[spike_frames]-xx[0])*1000,(Ycp[spike_frames]-yy[0])*1000,(Zcp[spike_frames]-zz[0])*1000,c='r',marker='o',alpha=0.05)
        ax3.patch.set_facecolor('w')
        ax3.view_init(15, 200) # Rotate plot for better view

        # set background color
        ax3.w_xaxis.set_pane_color((0., 0., 0., 0.4))
        ax3.w_yaxis.set_pane_color((0., 0., 0., 0.4))
        ax3.w_zaxis.set_pane_color((0., 0., 0., 0.4))

        # labeling
        ax3.set_title('3D whisker shape')
        ax3.set_xlabel('X(mm)',labelpad=10)
        ax3.set_ylabel('Y(mm)',labelpad=10)
        ax3.set_zlabel('Z(mm)',labelpad=2)

        # format numbers of labels
        ax3.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax3.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        # set axes limits based on min-max extent of the whisker
        ax3.set_xlim([bds['minx']*1000,bds['maxx']*1000])
        ax3.set_ylim([bds['miny']*1000, bds['maxy']*1000])
        ax3.set_zlim([bds['minz']*1000, bds['maxz']*1000])


        # ========= Mechanics and spike =================== #
        # get the indices in ms for the desired frame range.
        # Since the frame data is sampled at 300/500 fps and the filtered mechanics are at 1000fps, need to use the
        # time of the frames to index to the closest ms. This is approximate, and could be off by ~1ms in either direction.
        ms_idx = get_ms_idx(dat_3D, frame, t_range=T_RANGE)
        for ii,axis in enumerate([axMoment,axForce,axRotation]):
            # set up axes
            axis.cla()
            axis.axis('normal') # need to do this because of the previous call to equal axes?
            axis.set_xlim([ms_idx[0],ms_idx[1]])

            # plot mechanics for the given time.
            if ii==0:
                cmap = sns.color_palette('Reds_d',3)
                axis.set_prop_cycle(color=cmap)
                axis.plot(np.arange(ms_idx[0],ms_idx[1]),M[ms_idx[0]:ms_idx[1],:])
                axis.set_ylim([minM,maxM])
            elif ii==1:
                cmap = sns.color_palette('Blues_d',3)
                axis.set_prop_cycle(color=cmap)
                axis.plot(np.arange(ms_idx[0],ms_idx[1]),F[ms_idx[0]:ms_idx[1],:])
                axis.set_ylim([minF,maxF])
            elif ii==2:
                cmap = sns.color_palette('Greens_d',3)
                axis.set_prop_cycle(color=cmap)
                axis.plot(np.arange(ms_idx[0],ms_idx[1]),R[ms_idx[0]:ms_idx[1],:])
                axis.set_ylim([minR,maxR])

            # plot spikes for the given time
            sp_idx = np.logical_and(sp_1k>ms_idx[0],sp_1k<ms_idx[1])
            axis.vlines(sp_1k[sp_idx],axis.get_ylim()[0],axis.get_ylim()[1]/2,alpha=0.5)

            # plot a vertical line to indicate the current frame
            axis.vlines(dat_3D['frametimes'][frame][0] * 1000, axis.get_ylim()[0], axis.get_ylim()[1], colors='r', linestyles='dashed')

            # labelling
            if ii==0:
                axis.legend(['M$_x$','M$_y$','M$_z$','spikes'],loc=1,labelspacing=0)
                # axis.set_title('Moment over time')
                axis.set_xlabel('')
                axis.set_xticks([])
                axis.set_ylabel('Moment ($\mu$N-m)')
                sns.despine(ax=axis)
            elif ii==1:
                axis.legend(['F$_x$','F$_y$','F$_z$'],loc=1,labelspacing=0)
                # axis.set_title('Force')
                axis.set_xlabel('')
                axis.set_xticks([])
                axis.set_ylabel('Force ($\mu$N)')
                sns.despine(ax=axis)
            elif ii==2:
                axis.legend(['$\Delta\\theta$','$\Delta\phi$'],loc=1,labelspacing=0)
                # axis.set_title('Rotation')
                axis.set_xlabel('Time (ms)')
                axis.set_ylabel('Rotation')
                sns.despine(ax=axis)
        # save the image and iterate image number
        sns.despine(ax=axVid,left=True,bottom=True)
        plt.savefig(os.path.join(p_vid_save,f_vid_save+'{:05d}.png'.format(count)),dpi=300)
        count+=1

    audio = gen_spike_track(dat_3D,frames,fps=FPS)
    wavfile.write(os.path.join(p_vid_save,f_vid_save+'.wav'),44100,audio)
    subprocess.call(['ffmpeg',
                     '-f','image2',
                     '-r',str(FPS),
                     '-i',os.path.join(p_vid_save,f_vid_save+'%05d.png'),
                     '-i',os.path.join(p_vid_save,f_vid_save+'.wav'),
                     '-c:v','libx264',
                     '-c:a','libvo_aacenc',
                     '-y',
                     os.path.join(p_vid_save,f_vid_save+'.mp4')
                     ]
                    )
    # move all pngs
    new_subdir = os.path.join(p_vid_save,f_vid_save)
    os.makedirs(os.path.join(new_subdir))
    for item in os.listdir(p_vid_save):
        if item.endswith('.png') or item.endswith('.wav'):
            shutil.move(os.path.join(p_vid_save,item),os.path.join(new_subdir,item))

if __name__=='__main__':
    usage = "usage: %prog filename [options]"
    parser = OptionParser(usage)
    parser.add_option('-r','--ratnum',
                      dest='RATNUM',
                      default='2017_02',
                      type=str,
                      help='Rat number to grab from video')
    parser.add_option('-w','--whisker',
                      dest='WHISKER_ID',
                      default='B1',
                      type=str,
                      help='Whisker ID to grab')
    parser.add_option('-t','--trial',
                      dest='TRIAL',
                      default='t01',
                      type=str,
                      help='Trial number to grab')
    parser.add_option('-c','--cell',
                      dest='CELL_NUM',
                      default=0,
                      type=int,
                      help='cell nubmer to grab')
    parser.add_option('-s','--start',
                      dest='START_FRAME',
                      default=0,
                      type=int,
                      help='Index of the frame to start from')
    parser.add_option('-e','--stop',
                      dest='STOP_FRAME',
                      default=1,
                      type=int,
                      help='Index of the frame to end on')


    (options,args)=parser.parse_args()
    RATNUM = options.RATNUM
    WHISKER_ID = options.WHISKER_ID
    TRIAL = options.TRIAL
    CELL_NUM = options.CELL_NUM
    START_FRAME = options.START_FRAME
    STOP_FRAME = options.STOP_FRAME

    FRAME_STEP = 1
    HISTORY = 10
    FPS = 50
    T_RANGE = 250 # AMOUNT OF MECHANICAL DATA TO PLOT ON EITHER SIDE OF THE CURRENT TIME
    p_vid_save = r'C:\Users\guru\Desktop\test'
    p_vid_load = r'D:\VG3D\COMPRESSED'
    p_2D_data = r'K:\VG3D\tracked_2D\toMerge'
    p_3D_data = r'K:\VG3D\_rerun_with_pad\_deflection_trials\E3D'
    f_vid_save = '{}{}{}c{}_F{}F{}_example'.format(RATNUM,WHISKER_ID,TRIAL,CELL_NUM,START_FRAME,STOP_FRAME)
    main()
