# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tables

from brainDecTool.util import configParser
from brainDecTool.io import nifti as bdnifti
import util as vutil

# config parser
cf = configParser.Config('config')
data_dir = cf.get('base', 'path')

tf = tables.open_file(os.path.join(data_dir, 'VoxelResponses_subject1.mat'))
#tf.list_nodes

##-- load ROI mask
#l_v1_roi = tf.get_node('/roi/v1lh')[:]
#r_v1_roi = tf.get_node('/roi/v1rh')[:]
#l_v2_roi = tf.get_node('/roi/v2lh')[:]
#r_v2_roi = tf.get_node('/roi/v2rh')[:]
#l_v3_roi = tf.get_node('/roi/v3lh')[:]
#r_v3_roi = tf.get_node('/roi/v3rh')[:]
#l_v3a_roi = tf.get_node('/roi/v3alh')[:]
#r_v3a_roi = tf.get_node('/roi/v3arh')[:]
## merge ROIs in both hemisphere
#roi_mask = l_v1_roi + r_v1_roi*2 + l_v2_roi*3 + r_v2_roi*4 + \
#           l_v3_roi*5 + r_v3_roi*6 + l_v3a_roi*7 + r_v3a_roi*8
#nmask = vutil.convert2ras(roi_mask)
#bdnifti.save2nifti(np.around(nmask), 'nmask.nii.gz')

##-- make a mean response map as reference volume
#data = tf.get_node('/rt')[:]
## replace nan to zero
#data = np.nan_to_num(data)
#mean_data = np.mean(data, axis=1)
#vol = np.zeros((18, 64, 64))
#
#for i in range(data.shape[0]):
#    c = vutil.idx2coord(i)
#    vol[c[0], c[1], c[2]] = mean_data[i]
#
#nvol = vutil.convert2ras(vol)
#bdnifti.save2nifti(nvol, 'mean_vol.nii.gz')

tf.close()


#roi = tf.get_node('/roi/v1lh')[:].flatten()
#v1lh_idx = np.nonzero(roi==1)[0]
#v1lh_resp = data[v1lh_idx]

