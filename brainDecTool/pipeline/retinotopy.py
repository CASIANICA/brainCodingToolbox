# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
from brainDecTool.math import unit_vector

def coord2ecc(pos_mtx, image_size):
    """Return the distance to the image center.
    image_size : size (height,  width) of the stimulus.
    """
    if not isinstance(image_size, tuple):
        print 'A tuple is required for `image_size`'
        return

    center_pos = [image_size[0]*1.0/2-0.5, image_size[1]*1.0/2-0.5]
    row_num = pos_mtx.shape[0]
    cen = np.repeat([center_pos], row_num, axis=0)
    dist = np.linalg.norm(pos_mtx-cen, axis=1)
    return dist

def coord2angle(pos_mtx, image_size):
    """Return the angle given a coordinate in the image.
    image_size : size (height, width) of the stimulus.
    """
    if not isinstance(image_size, tuple):
        print 'A tuple is required for `image_size`'
        return

    std_pos = [0, image_size[1]*1.0/2-0.5]
    center_pos = [image_size[0]*1.0/2-0.5, image_size[1]*1.0/2-0.5]
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

