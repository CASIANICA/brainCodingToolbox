# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
from braincode.math import unit_vector

def coord2ecc(pos_mtx, img_size, max_visual_angle):
    """Return the distance to the image center in unit of degree (visual angle).
    image_size : size of the stimulus.
    """
    center_pos = [img_size/2.0-0.5, img_size/2.0-0.5]
    row_num = pos_mtx.shape[0]
    cen = np.repeat([center_pos], row_num, axis=0)
    dist = np.linalg.norm(pos_mtx-cen, axis=1)
    L = img_size / 2.0
    A = np.tan(max_visual_angle/360.0*np.pi)
    ecc = np.arctan(A*dist/L) / np.pi * 180
    return ecc

def coord2angle(pos_mtx, img_size):
    """Return the angle given a coordinate in the image.
    image_size : size of the stimulus.
    """
    std_pos = [0, img_size/2.0-0.5]
    center_pos = [img_size/2.0-0.5, img_size/2.0-0.5]
    std_vtr = np.array(std_pos) - np.array(center_pos)
    row_num = pos_mtx.shape[0]
    cen = np.repeat([center_pos], row_num, axis=0)
    vtr = pos_mtx - cen
    ang = np.zeros(row_num)
    for i in range(row_num):
        uvtr = unit_vector(vtr[i])
        usvtr = unit_vector(std_vtr)
        a = np.arccos(np.clip(np.dot(uvtr, usvtr), -1.0, 1.0))
        if vtr[i][1] < 0:
            ang[i] = 6.28 - a
        else:
            ang[i] = a
    return ang

def get_prf_size(pos_mtx, img_size, max_visual_angle):
    """Get pRF size based on pRF parameters.
    `pos_mtx` must be a n x 3 array, which the first two columns are pRF
    center position (row x column) in visual stimuli, and the 3rd column
    is the size of the pRF in pixels (i.e. FWHM).
    The size of pRF is returned in unit of degree (visual angle).
    """
    center_pos = [img_size/2.0-0.5, img_size/2.0-0.5]
    row_num = pos_mtx.shape[0]
    cen = np.repeat([center_pos], row_num, axis=0)
    center_dist = np.linalg.norm(pos_mtx[:, :2]-cen, axis=1)
    max_dist = center_dist + pos_mtx[:, 2]/2.0
    min_dist = center_dist - pos_mtx[:, 2]/2.0
    L = img_size / 2.0
    A = np.tan(max_visual_angle/360.0*np.pi)
    max_ecc = np.arctan(A*max_dist/L) / np.pi * 180
    min_ecc = np.arctan(A*min_dist/L) / np.pi * 180
    # due to the size was defined as FWHM in curve-fitting, the size of
    # pRF was be divided by 2.3548 to convert to sigma.
    return (max_ecc - min_ecc)/2.3548

