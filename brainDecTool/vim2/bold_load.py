# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tables

from brainDecTool.util import configParser

# config parser
cf = configParser.Config('config')
data_dir = cf.get('base', 'path')

tf = tables.open_file(os.path.join(data_dir, 'VoxelResponses_subjecr1.mat'))
tf.list_nodes

# merge v1 and v2 in both hemisphere
l_v1_roi = tf.get_node('/roi/v1lh')[:]
#r_v1_roi = tf.get_node('/roi/v1rh')[:]
#l_v2_roi = tf.get_node('/roi/v2lh')[:]
#r_v2_roi = tf.get_node('/roi/v2rh')[:]

# l_v1 range: 
#   i(A-P): 0 ~ 14
#   j(I-S): 11 ~ 28
#   k(R-L): 31 ~ 39
# r_v1 range:
#   i: 0 ~ 17
#   j: 13 ~ 27
#   k: 23 ~ 31
tmp_l_v1 = np.zeros((l_v1_roi.shape[]))

data = tf.get_node('/rt')[:]
roi = tf.get_node('/roi/v1lh')[:].flatten()
v1lh_idx = np.nonzero(roi==1)[0]
v1lh_resp = data[v1lh_idx]
tf.close()

