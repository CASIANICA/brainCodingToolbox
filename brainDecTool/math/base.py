# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
from skimage.measure import block_reduce
from skimage.transform import resize
from joblib import Parallel, delayed

def corr2_coef(A, B, mode='full'):
    """Row-wise Correlation Coefficient calculation for two 2D arrays.
    Input 2D array A (m by d), array B (n by d), in `full` model, the function
    would return a correlation matrix (m by n); in `pair` model, array A and
    B must have identical shape (m == n), and a correlation array for each
    pair of vector would be returned.

    """
    if mode=='full':
        # Row-wise mean of input arrays & subtract from input arrays themselves
        A_mA = A - A.mean(1)[:, None]
        B_mB = B - B.mean(1)[:, None]
        # Sum of squares across rows
        ssA = (A_mA**2).sum(1)
        ssB = (B_mB**2).sum(1)
        # Finally get corr coef
        return np.dot(A_mA, B_mB.T)/np.sqrt(np.dot(ssA[:, None], ssB[None]))
    elif mode=='pair':
        norm_A = (A - A.mean(1)[:, None]) / A.std(1)[:, None]
        norm_B = (B - B.mean(1)[:, None]) / B.std(1)[:, None]
        norm_A = np.nan_to_num(norm_A)
        norm_B = np.nan_to_num(norm_B)
        return np.einsum('ij, ij->i', norm_A, norm_B)/A.shape[1]

def parallel_corr2_coef(A, B, filename, block_size=32, n_jobs=8):
    """Compute row-wise correlation coefficient for two 2D arrays in a
    parallel computing approach.
    Array `B` would be divided into several blocks each containing
    `block_size` rows for one parallel computing iteration.
    
    Usage
    -----
    cross_modal_corr(A, B, filename, block_size=32, n_jobs=8)
    
    Return
    ------
    A row-wise correlation matrix saved in `filename`. For example, if
    the size of `A` is (p, n), and the size of `B` is (q, n), the
    size of return matrix is (p, q).

    Note
    ----
    'block_size' : the number of rows in `B` processed in one iter.
    """
    # to reduce memory usage, we compute Pearson's r iteratively
    A_size = A.shape[0]
    B_size = B.shape[0]
    corr_mtx = np.memmap(filename, dtype='float16', mode='w+',
                         shape=(A_size, B_size))
    print 'Compute row-wise correlation ...'
    # parallelize the corr computation
    Parallel(n_jobs=n_jobs)(delayed(pcorr2_sugar)(A, B, corr_mtx, i,
                            block_size) for i in range(B_size/block_size))
    narray = np.nan_to_num(np.array(corr_mtx))
    np.save(filename, narray)

def pcorr2_sugar(A, B, output, i, block_size):
    """Sugar function for parallel computing."""
    print 'Iter %s' %(i)
    if (i+1)*block_size > B.shape[0]:
        output[:, i*block_size:] = corr2_coef(A, B[i*block_size:, :])
    else:
        output[:, i*block_size:(i+1)*block_size] = corr2_coef(A,
                    B[i*block_size:(i+1)*block_size, :])

def unit_vector(vector):
    """Return the unit vector of the input."""
    return vector / np.linalg.norm(vector)

def down_sample(image, block_size, cval=0):
    """Down-sampling an input image, and the unit of down-sample is specified
    with `block_size`.
    `cval` : Constant padding value is image is not perfectly divisible by
    the block size.

    Example
    -------
    image shape : 10, 10, 8
    block_size : 2, 2, 1
    the down_sample(image, block_size=(2, 2, 1)) would return an image which 
    shape is (5, 5, 8).
    """
    return block_reduce(image, block_size, func=np.mean, cval=cval)

def img_resize(img, out_dim):
    """Resize image to `out_dim`.
    img is a 3d array which first 2 dim corresponding image size
    out_dim is a tuple containing resized image size
    """
    im_min, im_max = img.min(), img.max()
    im_std = (img - im_min) / (im_max - im_min)
    resized_im = resize(im_std, out_dim, order=1)
    resized_im = resized_im * (im_max - im_min) + im_min
    return resized_im

def time_lag_corr(x, y, maxlag):
    """Calculate cross-correlation between x and a lagged y.
    `x` and `y` are two 1-D vector, `maxlag` refers to the maximum lag value.

    formula
    -------
    c_{xy}[k] = sum_n x[n] * y[n+k]
    k : 0 ~ (maxlag-1)

    """
    c = np.zeros(maxlag)
    y = np.array(y)
    for i in range(maxlag):
        lagy = np.array(y[i:].tolist()+[0]*i)
        c[i] = np.correlate(x, lagy) / len(x)
    return c

