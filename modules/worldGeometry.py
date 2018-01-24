import neoUtils
import numpy as np
from sklearn import mixture
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import quantities as pq

def get_delta_angle(blk):
    '''
    This function returns the changes in world angle with respect to the first frame of contact.
    This should give us an estimate of how much the whisker is rotating in the follicle
    :param blk: a neo block
    
    :return th_contact, phie_contacts : a [t x n] matrix where t is the number of time samples in the longest contact and n is the number of contacts
    '''
    PHIE = neoUtils.get_var(blk,'PHIE')
    TH = neoUtils.get_var(blk, 'TH')
    use_flags = neoUtils.concatenate_epochs(blk,epoch_idx=-1)
    phie_contacts =  neoUtils.get_analog_contact_slices(PHIE,use_flags).squeeze()
    th_contacts = neoUtils.get_analog_contact_slices(TH, use_flags).squeeze()

    d = np.sqrt(phie_contacts ** 2 + th_contacts ** 2)
    use = np.invert(np.all(np.isnan(d), axis=0)) # remove all nan slices
    return(th_contacts[:,use],phie_contacts[:,use])


def center_angles(th_contacts,ph_contacts):
    for th,ph in zip(th_contacts,ph_contacts):
        if np.all(np.isnan(th)) or np.all(np.isnan(ph)):
            continue

        first_index = np.min((
            np.where(np.isfinite(th))[0][0],
            np.where(np.isfinite(ph))[0][0])
        )

        th-=th[first_index]
        ph-=ph[first_index]


def get_max_angular_displacement(th_contacts,phie_contacts):
    '''
    Finds the point of maximal displacement in the theta-phi space and returns the 
    value of theta and phi for that point, for all contacts.
    
    :param th_contacts:     a 1D numpy vector of theta values 
    :param phie_contacts:   a 1D numpy vector of Phi values
    
    :return th_max,phi_max: 1D vectors of the values of Theta and Phi and maximal angular displacement
    '''

    # get maximal displacement and index when that occurs
    d = np.hypot(phie_contacts,th_contacts)
    idx = np.nanargmax(d,axis=0)

    # Loop through each contact to extract the point of maximal displacement
    phi_max = [phie_contacts[x, contact_idx] for contact_idx, x in enumerate(idx)]
    th_max = [th_contacts[x, contact_idx] for contact_idx, x in enumerate(idx)]

    # cast to numpy array
    phi_max = np.array(phi_max)
    th_max = np.array(th_max)

    return th_max,phi_max


def norm_angles(th_max,ph_max):
    '''
    Map the theta and phi vectors to the unit circle. This is useful in the classifier.
    
    :param th_max:      The values of theta at maximal deflection for each contact
    :param ph_max:      The values of phi at maximal deflection for each contact
    
    :return th_norm,ph_norm: Normalized values of theta and phi such that theta/phi lie on the unit circle 
    '''

    X = np.concatenate([th_max[:, np.newaxis], ph_max[:, np.newaxis]], axis=1)
    X_norm = np.linalg.norm(X,axis=1)
    return(th_max/X_norm,ph_max/X_norm)


def get_radial_distance_group(blk,plot_tgl=False):
    S = neoUtils.get_var(blk,'S')
    use_flags = neoUtils.concatenate_epochs(blk,-1)
    S_contacts = neoUtils.get_analog_contact_slices(S,use_flags)
    S_med = np.nanmedian(S_contacts,axis=0)[:,np.newaxis]

    clf3 = mixture.GaussianMixture(n_components=3,n_init=100)
    clf2 = mixture.GaussianMixture(n_components=2,n_init=100)
    clf3.fit(S_med)
    clf2.fit(S_med)
    if clf2.aic(S_med)<clf3.aic(S_med):
        n_clusts=2
        idx = clf2.predict(S_med)
    else:
        n_clusts=3
        idx = clf3.predict(S_med)

    S_clusts = []
    for ii in xrange(n_clusts):
        S_clusts.append(np.nanmedian(S_med[idx==ii]))
    ordering = np.argsort(S_clusts)
    idx = np.array([np.where(x == ordering)[0][0] for x in idx])
    S_clusts.sort()
    if np.any(np.isnan(S_clusts)):
        return(-1)
    if plot_tgl:
        cc = sns.color_palette("Blues", n_clusts+3)
        for ii in xrange(n_clusts):
            sns.distplot(S_med[idx==ii],color = cc[ii+3])
        ax = plt.gca()
        ax.set_ylabel('Number of contacts')
        ax.set_xlabel('Arclength at contact (m)')
        ax.grid('off', axis='x')
        ax.set_title('{}'.format(neoUtils.get_root(blk,0)))

    return(idx)


def get_contact_direction(blk,plot_tgl=True):
    '''
    Classifies all the contacts into one of 8 directions.
    
    Orders the groups such that the first contacts are at ~0 radians, and the direction indicies
    increase as the direction increases from 0 to 2Pi 
    
    Note - the groups roughly increase by 45 degrees, but not explicitly so
    
    :param blk:         A neo block
    
    :param plot_tgl:    A switch as to whether you want to plot the deflections and their groups for inspection.
                        Default = True
                        
    :return idx:   idx: 
                     Index for every contact as to which group 
                     that contact belongs. Index 0 is the first direction, 
                     index 1 is the next clockwise group, etc...
                    med_angle: median angle in the theta/phi centered space
    
    '''
    def reduce_deflection(blk, num_pts):
        '''
        Grabs a subset of theta/phi points from a deflection.
        Probably not useful. Defaults to error

        :param blk: 
        :param num_pts: 
        :param pad: 
        :return: 
        '''

        # if True:
        #     raise Exception('This function is not finished, and is likely not useful. NEB 20180109')

        th_contacts, ph_contacts = get_delta_angle(blk)
        center_angles(th_contacts, ph_contacts)

        X = np.empty([th_contacts.shape[1], num_pts * 2])
        for ii in xrange(ph_contacts.shape[-1]):
            th = th_contacts[:, ii]
            ph = ph_contacts[:, ii]

            th = th[np.isfinite(th)]
            ph = ph[np.isfinite(ph)]

            th_pts = th[np.linspace(0, len(th) - 1, num_pts).astype('int')]
            ph_pts = ph[np.linspace(0, len(ph) - 1, num_pts).astype('int')]
            # th_pts = th_pts[pad / 2:-pad / 2]
            # ph_pts = ph_pts[pad / 2:-pad / 2]

            X[ii, :] = np.concatenate([th_pts, ph_pts])
        return (X)
    # init output index vector
    use_flags = neoUtils.concatenate_epochs(blk, epoch_idx=-1)
    n_contacts = len(use_flags)
    if n_contacts<50:
        return(-1,-1)
    direction_index = np.empty(n_contacts,dtype='int');
    direction_index[:] = np.nan

    # get contact angles and zero them to the start of contact
    th_contacts,ph_contacts = get_delta_angle(blk)
    center_angles(th_contacts,ph_contacts)

    # get max angles
    th_max,ph_max = get_max_angular_displacement(th_contacts,ph_contacts)

    # get PCA decomp
    X = reduce_deflection(blk,10)
    pca = sklearn.decomposition.PCA()
    Y = pca.fit_transform(X)
    Y1,Y2 = norm_angles(Y[:,0],Y[:,1])
    Y = np.concatenate([Y1[:,np.newaxis],Y2[:,np.newaxis]],axis=1)

    # group angles into a design matrix and get the direction of the deflection [-pi:pi]
    projection_angle = np.arctan2(np.deg2rad(ph_max),np.deg2rad(th_max))[:,np.newaxis]

    good_contacts = np.all(np.isfinite(Y), axis=1)
    Y = Y[good_contacts,:] # remove nan groups
    projection_angle = projection_angle[good_contacts].ravel()

    # Cluster the groups
    clf = mixture.GaussianMixture(n_components=8,n_init=100)
    clf.fit(Y)
    idx = clf.predict(Y)

    # get the median angles and sort with the first direction stimulated as zero
    med_angle = []
    for ii in xrange(8):
        med_angle.append(np.nanmedian(projection_angle[idx==ii]))
    med_angle = np.array(med_angle)

    # sort the group indices such that the first deflection angle is 0, and they increase from there
    new_idx = np.argsort(med_angle)
    new_idx = [np.where(x==new_idx)[0][0] for x in idx]

    first_idx = scipy.stats.mode(new_idx[:40]).mode
    new_idx-=first_idx
    new_idx[new_idx<0]+=8

    # get the new median angles (in the centered theta/phi space)
    med_angle = []
    for ii in xrange(8):
        med_angle.append(np.nanmedian(projection_angle[new_idx==ii]))
    med_angle = np.array(med_angle)

    # plotting
    if plot_tgl:
        cc = sns.color_palette("husl", 8)
        for ii in xrange(X.shape[0]):
            plt.plot(th_contacts[:,ii],ph_contacts[:,ii],'.-',color=cc[new_idx[ii]],alpha=0.3)
        for ii in xrange(8):
            plt.plot(np.cos(med_angle[ii]),np.sin(med_angle[ii]),'o',markersize=10,color=cc[ii],markeredgecolor='k',markeredgewidth=1)

    # map the used indexes to the output so that we dont misalign contacts with their group index
    direction_index[good_contacts]=new_idx
    return(direction_index,med_angle)


def get_onset_velocity(blk,plot_tgl=False):
    '''
    Get the onset and offset velocities for each deflection 
    with respect to the angular motion of the base (TH,PHIE)
    
    :param blk: 
    :param plot_tgl: 
    :return: 
    '''
    use_flags = neoUtils.concatenate_epochs(blk, -1)
    durations = use_flags.durations

    th, ph = get_delta_angle(blk)
    center_angles(th,ph)
    th_max,ph_max = get_max_angular_displacement(th,ph)

    # distance from initial point at all times
    D = np.hypot(th, ph)*pq.rad

    # time of max displacement
    idx = np.nanargmax(D, axis=0)*pq.ms

    # max distance from initial point
    D_max = np.hypot(th_max, ph_max)*pq.deg

    V_onset = D_max / idx * pq.ms
    V_offset = D_max / (durations-idx) * pq.ms

    if plot_tgl:

        time_lim = np.percentile(durations, 90)
        cmap = sns.cubehelix_palette(as_cmap=True,dark=0.1,light=0.8)
        color_to_plot = (V_onset/np.nanmax(V_onset[np.isfinite(V_onset)])).magnitude
        color_to_plot[np.isinf(color_to_plot)]=0
        f = plt.figure()
        for ii in xrange(th.shape[-1]):
            # Plot Theta
            ax = f.add_subplot(311)
            plt.plot(th[:,ii],color=cmap(color_to_plot[ii]),alpha=.2)
            ax.set_title(r'$\theta$')
            ax.set_ylabel('Degrees')
            ax.set_xlim(0,time_lim)

            # Plot Phi
            ax = f.add_subplot(312)
            plt.plot(ph[:, ii], color=cmap(color_to_plot[ii]), alpha=.2)
            ax.set_title(r'$\phi$')
            ax.set_ylabel('Degrees')
            ax.set_xlim(0, time_lim)

            # Plot sqrt(th**2+phi**2)
            ax = f.add_subplot(313)
            plt.plot(D[:, ii], color=cmap(color_to_plot[ii]), alpha=.2)
            ax.set_title(r'angular displacement')
            ax.set_xlabel('time (ms)')
            ax.set_ylabel('Degrees')
            ax.set_xlim(0, time_lim)

        f.suptitle('{}'.format(neoUtils.get_root(blk,0)))
        f.tight_layout(rect=[0, 0.03, 1, 0.95])

    return(V_onset,V_offset)


def CP_to_world(blk):
    '''
    Transforms the contact point back into the world reference frame,
    accounting for both the rotation and bending of the whisker.
    
    :param blk: 
    :return CP_world: 
    '''
    CP = neoUtils.get_var(blk,'CP',keep_neo=False)[0]
    PH = neoUtils.get_var(blk,'PHIE',keep_neo=False)[0]
    PH = np.deg2rad(PH)
    TH = neoUtils.get_var(blk, 'TH',keep_neo=False)[0]
    TH =np.deg2rad(TH)
    Z = neoUtils.get_var(blk,'ZETA',keep_neo=False)[0]
    # Z = np.deg2rad(Z)
    BP = neoUtils.get_var(blk, 'BPm',keep_neo=False)[0]
    cbool = neoUtils.get_Cbool(blk)

    CP_world = np.empty_like(CP); CP_world[:]=np.nan
    def RX(theta):
        c = np.cos(theta)[0]
        s = np.sin(theta)[0]
        return(np.array([[1,0,0],[0,c,-s],[0,s,c]]))
    def RY(theta):
        c = np.cos(theta)[0]
        s = np.sin(theta)[0]
        return(np.array([[c,0,s],[0,1,0],[-s,0,c]]))
    def RZ(theta):
        c = np.cos(theta)[0]
        s = np.sin(theta)[0]
        return(np.array([[c,-s,0],[s,c,0],[0,0,1]]))
    for ii in xrange(CP.shape[0]):
        if cbool[ii]:
            ROT = np.linalg.multi_dot([RX(-Z[ii]),RY(-PH[ii]),RZ(-TH[ii])])
            CP_world[ii,:] = np.dot(np.linalg.inv(ROT),CP[ii,:][:,np.newaxis]).T
            CP_world[ii,:]+=BP[ii,:]
    return CP_world
