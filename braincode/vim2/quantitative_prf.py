# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import matplotlib.pylab as plt

from braincode.vim2.dataio import load_prf_data

root_dir = r'/Users/sealhuang/project/brainCoding'
prf_dir = os.path.join(root_dir, 'subjects', 'vS1', 'prf',
                       'gaussian_kernel', 'v1lh')

prf_data = load_prf_data(prf_dir)

# select significantly predicted voxels
thr = 0.25
sel_vxl = prf_data['val_corr']>=thr

# relation between type of pooling size and eccentricity
model_type = prf_data['sel_model_idx'] / 1024
plt.scatter(prf_data['ecc'][sel_vxl], model_type[sel_vxl], 5);plt.show()
np.corrcoef(prf_data['ecc'][sel_vxl], model_type[sel_vxl])

# relation between weight sum of different spatial frequencies and eccentricity
wt_sum = np.zeros((490, 5))
for i in range(5):
    wt_sum[:, i] = np.sum(prf_data['paras'][:, (i*8):(i*8+8)], axis=1)
    np.corrcoef(prf_data['ecc'][sel_vxl], wt_sum[sel_vxl, i])
    plt.scatter(prf_data['ecc'][sel_vxl], wt_sum[sel_vxl, i]);plt.show()


