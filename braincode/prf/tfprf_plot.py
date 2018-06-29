# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

db_dir = r'/Users/sealhuang/project/brainCoding/subjects/vim1_S1/prf/v1'

#-- plot validation r-square across stages
#dl_val_r2 = np.load(os.path.join(db_dir, 'dl_prf_val_r2.npy'))
#dl_refine_val_r2 = np.load(os.path.join(db_dir, 'dl_prf_refine_val_r2.npy'))
#plt.scatter(dl_val_r2, dl_refine_val_r2, 4)
#plt.plot([-0.2, 0.5], [-0.2, 0.5], 'r--')
#plt.xlim(-0.35, 0.55)
#plt.ylim(-0.35, 0.55)
#plt.xlabel('validation r-square of CNN-pRF')
#plt.ylabel('validation r-square of refined CNN-pRF')
#plt.savefig(os.path.join(db_dir, 'validation_r2_across_stages.png'))
#plt.close()

#-- plot test r-square across stages
#dl_test_r2 = np.load(os.path.join(db_dir, 'dl_prf_test_r2.npy'))
#dl_refine_test_r2 = np.load(os.path.join(db_dir, 'dl_prf_refine_test_r2.npy'))
#plt.scatter(dl_test_r2, dl_refine_test_r2, 4)
#plt.plot([0, 0.5], [0, 0.5], 'r--')
#plt.xlim(-0.05, 0.6)
#plt.ylim(-0.05, 0.6)
#plt.xlabel('test r-square of CNN-pRF')
#plt.ylabel('test r-square of refined CNN-pRF')
#plt.savefig(os.path.join(db_dir, 'test_r2_across_stages.png'))
#plt.close()

#-- plot test r-square across stages
#reg_test_r2 = np.load(os.path.join(db_dir, 'reg_prf_test_r2.npy'))
#dl_refine_test_r2 = np.load(os.path.join(db_dir, 'dl_prf_refine_test_r2.npy'))
#plt.scatter(reg_test_r2, dl_refine_test_r2, 4)
#plt.plot([0, 0.5], [0, 0.5], 'r--')
#plt.xlim(-0.1, 0.6)
#plt.ylim(-0.1, 0.6)
#plt.xlabel('test r-square of Gaussian-pRF')
#plt.ylabel('test r-square of CNN-pRF')
#plt.savefig(os.path.join(db_dir, 'test_r2_across_methods.png'))
#plt.close()

#-- plot histogram of difference between Gaussian- and CNN-pRF
reg_test_r2 = np.load(os.path.join(db_dir, 'reg_prf_test_r2.npy'))
dl_test_r2 = np.load(os.path.join(db_dir, 'dl_prf_refine_test_r2.npy'))
thres = 0.1
sel_idx = []
for i in range(reg_test_r2.shape[0]):
    if reg_test_r2[int(i)]>=thres or dl_test_r2[int(i)]>=thres:
        sel_idx.append(i)
# select 386 voxels, in which 230 voxels show larger fitting performance on
# CNN-pRF method
diff = dl_test_r2[sel_idx] - reg_test_r2[sel_idx]
plt.hist(diff, 35)
plt.plot([0, 0], [0, 30], 'r--')
plt.xlabel('difference of fitting performance between CNN- and Gaussian-pRF\n(r-square on test dataset)')
plt.ylabel('voxel#')
plt.savefig(os.path.join(db_dir, 'hist_diff_between_methods.png'))
plt.close()

