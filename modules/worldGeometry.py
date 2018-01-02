import neo
import neoUtils
import sys
import os
import numpy as np
from sklearn import mixture
from sklearn import decomposition
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

    # phie_contacts -= phie_contacts[0, :]
    # th_contacts -= th_contacts[0, :]
    return th_contacts,phie_contacts

def get_max_angular_displacement(th_contacts,phie_contacts):
    d = np.sqrt(phie_contacts**2+th_contacts**2)
    max_d = np.nanmax(d,axis=0)
    idx = np.nanargmax(d,axis=0)

    phi_max = []
    th_max = []
    for ii in xrange(phie_contacts.shape[1]):
        phi_max.append(phie_contacts[idx[ii],ii])
        th_max.append(th_contacts[idx[ii], ii])
    phi_max = np.array(phi_max)
    th_max = np.array(th_max)

    # md = np.arctan2(phi_max,th_max)
    return th_max,phi_max

def get_r_contact(blk):
    r = neoUtils.get_var(blk,'Rcp')
    use_flags = neoUtils.concatenate_epochs(blk, epoch_idx=-1)
    r_contacts =  neoUtils.get_analog_contact_slices(r,use_flags).squeeze()
    return(np.nanmean(r_contacts,axis=0))

def reduce_deflection(blk,num_pts,buffer=5):
    th_contacts,ph_contacts = get_delta_angle(blk)
    th_contacts -= th_contacts[0,:]
    ph_contacts -= ph_contacts[0, :]

    X = np.empty([th_contacts.shape[1],num_pts*2])
    for ii in xrange(ph_contacts.shape[-1]):
        th = th_contacts[:, ii]
        ph = ph_contacts[:, ii]

        th = th[np.isfinite(th)]
        ph = ph[np.isfinite(ph)]

        th_pts = th[np.linspace(0,len(th)-1,num_pts+buffer).astype('int')]
        ph_pts = ph[np.linspace(0, len(ph)-1, num_pts+buffer).astype('int')]
        th_pts = th_pts[buffer/2:-buffer/2]
        ph_pts = ph_pts[buffer/2:-buffer/2]

        X[ii,:] = np.concatenate([th_pts,ph_pts])
    return(X)

def get_contact_direction(blk,plot_tgl=True):
    '''
    
    :param blk: 
    :param plot_tgl: 
    :return: 
    '''
    th_contacts,ph_contacts = get_delta_angle(blk)
    th_contacts -= th_contacts[0, :]
    ph_contacts -= ph_contacts[0, :]
    th_med = np.nanmedian(th_contacts,axis=0)
    ph_med = np.nanmedian(ph_contacts, axis=0)

    th_mean = np.nanmean(th_contacts, axis=0)
    ph_mean = np.nanmean(ph_contacts, axis=0)

    th_max,ph_max = get_max_angular_displacement(th_contacts,ph_contacts)

    d1 = np.arctan2(np.deg2rad(ph_max),np.deg2rad(th_max))[:,np.newaxis]
    d2 = np.arctan2(np.deg2rad(ph_mean), np.deg2rad(th_mean))[:, np.newaxis]
    d3 = np.arctan2(np.deg2rad(ph_med), np.deg2rad(th_med))[:, np.newaxis]

    X = np.concatenate([d1,d2,d3],axis=1)
    X = np.unwrap(X,axis=0,discont = np.pi*1.2)


    X = np.concatenate([X,np.arange(d.shape[0])[:,np.newaxis]],axis=1)
    clf = mixture.GaussianMixture(n_components=8,n_init=100)

    clf.fit(X)
    idx = clf.predict(X)


    # get teh angles
    med_angle = []
    for ii in xrange(8):
        med_angle.append(np.nanmedian(d[idx==ii]))
    med_angle = np.array(med_angle)

    first_angle = np.nanmean(d[:10])


    if plot_tgl:
        cc = sns.color_palette("Set2", 8)
        for ii in xrange(X.shape[0]):
            plt.plot(th_contacts[:,ii],ph_contacts[:,ii],'.-',color=cc[idx[ii]],alpha=0.3)
        for ii in xrange(8):
            plt.plot(np.cos(med_angle[ii]),np.sin(med_angle[ii]),'o',markersize=10,color=cc[ii],markeredgecolor='k',markeredgewidth=1)

    return(idx)


