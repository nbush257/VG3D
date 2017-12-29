import neo
import neoUtils
import sys
import os
import numpy as np

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

    phie_contacts -= phie_contacts[0, :]
    th_contacts -= th_contacts[0, :]
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

