# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tables
from scipy.misc import imsave
import cv2

from brainDecTool.util import configParser


def mat2png(stimulus, prefix_name):
    """Comvert stimulus from mat to png format."""
    for i in range(stimulus.shape[0]):
        x = stimulus[i, :]
        new_x = np.zeros((x.shape[1], x.shape[2], 3), dtype=np.uint8)
        new_x[..., 0] = x[0, ...]
        new_x[..., 1] = x[1, ...]
        new_x[..., 2] = x[2, ...]
        new_x = np.transpose(new_x, (1, 0, 2))
        file_name = prefix_name + '_' + str(i+1) + '.png'
        imsave(file_name, new_x)

def get_optical_flow(stimulus, prefix_name):
    """Calculate dense optical flow from stimuli sequence."""
    img_w, img_h = stimulus.shape[2], stimulus.shape[3]
    mag = np.zeros((img_h, img_w, stimulus.shape[0]), dtype=np.float32)
    ang = np.zeros((img_h, img_w, stimulus.shape[0]), dtype=np.float32)
    for i in range(1, stimulus.shape[0]):
        if i==1:
            pim = np.zeros((img_w, img_h, 3), dtype=np.uint8)
            pim[..., 0] = stimulus[0, 0, ...]
            pim[..., 1] = stimulus[0, 1, ...]
            pim[..., 2] = stimulus[0, 2, ...]
            pim = np.transpose(pim, (1, 0, 2))
            prvs = cv2.cvtColor(pim, cv2.COLOR_BGR2GRAY)
        nim = np.zeros((img_w, img_h, 3), dtype=np.uint8)
        nim[..., 0] = stimulus[i, 0, ...]
        nim[..., 1] = stimulus[i, 1, ...]
        nim[..., 2] = stimulus[i, 2, ...]
        nim = np.transpose(nim, (1, 0, 2))
        next = cv2.cvtColor(nim, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, 0.5, 3, 15,
                                            3, 5, 1.2, 0)
        mtmp, atmp = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag[..., i] = mtmp
        ang[..., i] = atmp
        prvs = next
    np.save(prefix_name+'_opticalflow_mag.npy', mag)
    np.save(prefix_name+'_opticalflow_ang.npy', ang)

if __name__ == '__main__':
    """Main function."""
    # config parser
    cf = configParser.Config('config')
    data_dir = cf.get('base', 'path')
    tf = tables.open_file(os.path.join(data_dir, 'Stimuli.mat'))
    #tf.listNodes
    stimulus = tf.get_node('/st')[:]
    tf.close()

    # convert mat to png
    #mat2png(stimulus, 'train')

    # calculate dense optical flow
    get_optical_flow(stimulus, 'train')
