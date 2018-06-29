# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

db_dir = r'/Users/sealhuang/project/brainCoding/subjects/vim1_S1/prf/v1'

# plot validation r-square across stages
dl_val_r2 = np.load(os.path.join(db_dir, 'dl_prf_val_r2.npy'))
dl_refine_val_r2 = np.load(os.path.join(db_dir, 'dl_prf_refine_val_r2.npy'))
plt.scatter(dl_val_r2, dl_refine_val_r2, 4)
plt.plot([-0.2, 0.5], [-0.2, 0.5], 'r--')
plt.xlim(-0.35, 0.55)
plt.ylim(-0.35, 0.55)
plt.xlabel('validation r-square of CNN-pRF')
plt.ylabel('validation r-square of refined CNN-pRF')
plt.savefig(os.path.join(db_dir, 'validation_r2_across_stages.png'))
plt.close()

# plot test r-square across stages
dl_test_r2 = np.load(os.path.join(db_dir, 'dl_prf_test_r2.npy'))
dl_refine_test_r2 = np.load(os.path.join(db_dir, 'dl_prf_refine_test_r2.npy'))
plt.scatter(dl_test_r2, dl_refine_test_r2, 4)
plt.plot([0, 0.5], [0, 0.5], 'r--')
plt.xlim(-0.05, 0.6)
plt.ylim(-0.05, 0.6)
plt.xlabel('test r-square of CNN-pRF')
plt.ylabel('test r-square of refined CNN-pRF')
plt.savefig(os.path.join(db_dir, 'test_r2_across_stages.png'))
plt.close()

