import neo
import neoUtils
import sys
import os
import numpy as np
from sklearn import mixture
import scipy
import seaborn as sns
import quantities as pq
import matplotlib.pyplot as plt


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

    return th_contacts,phie_contacts

def get_max_angular_displacement(th_contacts,phie_contacts):
    '''
    Finds the point of maximal displacement in the theta-phi space and returns the 
    value of theta and phi for that point, for all contacts.
    
    :param th_contacts:     a 1D numpy vector of theta values 
    :param phie_contacts:   a 1D numpy vector of Phi values
    
    :return th_max,phi_max: 1D vectors of the values of Theta and Phi and maximal angular displacement
    '''

    # get maximal displacement and index when that occurs
    d = np.sqrt(phie_contacts**2+th_contacts**2)
    idx = np.nanargmax(d,axis=0)

    # Loop through each contact to extract the point of maximal displacement
    phi_max = [phie_contacts[x, contact_idx] for contact_idx, x in enumerate(idx)]
    th_max = [th_contacts[x, contact_idx] for contact_idx, x in enumerate(idx)]

    # cast to numpy array
    phi_max = np.array(phi_max)
    th_max = np.array(th_max)

    return th_max,phi_max


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

def norm_angles(th_max,ph_max):
    X = np.concatenate([th_max[:, np.newaxis], ph_max[:, np.newaxis]], axis=1)
    X_norm = np.linalg.norm(X,axis=1)
    return(th_max/X_norm,ph_max/X_norm)

def get_contact_direction(blk,plot_tgl=True):
    # TODO: make this a constant 2pi/8 regions centered around the first contact?
    '''
    
    :param blk: 
    :param plot_tgl: 
    :return: 
    '''
    th_contacts,ph_contacts = get_delta_angle(blk)
    th_contacts -= th_contacts[0, :]
    ph_contacts -= ph_contacts[0, :]

    th_max,ph_max = get_max_angular_displacement(th_contacts,ph_contacts)
    th_norm,ph_norm = norm_angles(th_max,ph_max)

    d = np.arctan2(np.deg2rad(ph_max),np.deg2rad(th_max))[:,np.newaxis]
    X = np.concatenate([th_norm[:,np.newaxis],ph_norm[:,np.newaxis]],axis=1)
    t = np.arange(d.shape[0])[:,np.newaxis]/float(d.shape[0])
    # X = np.concatenate([X,t],axis=1)

    clf = mixture.GaussianMixture(n_components=8,n_init=100)
    clf.fit(X)
    idx = clf.predict(X)

    # get the median angles and sort with the first direction stimulated as zero
    med_angle = []
    for ii in xrange(8):
        med_angle.append(np.nanmedian(d[idx==ii]))
    med_angle = np.array(med_angle)

    new_idx = np.argsort(med_angle)
    new_idx = [np.where(x==new_idx)[0][0] for x in idx]

    first_idx = scipy.stats.mode(new_idx[:40]).mode
    new_idx-=first_idx
    new_idx[new_idx<0]+=8

    med_angle = []
    for ii in xrange(8):
        med_angle.append(np.nanmedian(d[new_idx==ii]))
    med_angle = np.array(med_angle)

    if plot_tgl:
        cc = sns.color_palette("husl", 8)
        for ii in xrange(X.shape[0]):
            plt.plot(th_contacts[:,ii],ph_contacts[:,ii],'.-',color=cc[new_idx[ii]],alpha=0.3)
        for ii in xrange(8):
            plt.plot(np.cos(med_angle[ii]),np.sin(med_angle[ii]),'o',markersize=10,color=cc[ii],markeredgecolor='k',markeredgewidth=1)


    return(new_idx,med_angle)


