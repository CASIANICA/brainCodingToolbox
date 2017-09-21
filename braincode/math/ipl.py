# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
Image processing library (ipl)
"""

import numpy as np

def sugar_func(t):
    """Util function for rgb2cielab"""
    n = np.zeros_like(t)
    hithres = t>0.008856
    lowthres = t<=0.008856
    n[hithres] = np.power(t[hithres], 1/3.0)
    n[lowthres] = 7.787 * t[lowthres] + 16 / 116.0
    return n

def rgb2cielab(rgb_img):
    """Convert an image from RGB space to CIEL*A*B* space.
    Output channel sequence: L* A* B*.
    """
    #Conversion Matrix
    matrix = [[0.412453, 0.357580, 0.180423],
              [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]]

    # RGB values lie between 0 to 1.0
    img_d = rgb_img.shape[0]
    img_w = rgb_img.shape[1]
    rgb = np.transpose(rgb_img, (2, 0, 1)).astype('float')/255
    cie = np.dot(matrix, rgb.reshape(3, img_d*img_w));
    cie = cie.reshape(3, img_d, img_w)

    cie[0, ...] = cie[0, ...] / 0.950456;
    cie[2, ...] = cie[2, ...] / 1.088754; 

    # Calculate the L
    L = np.zeros_like(cie[1, ...])
    hithres = cie[1, ...]>0.008856
    lowthres = cie[1, ...]<=0.008856
    L[hithres] = 116 * np.power(cie[1, hithres], 1/3.0) - 16.0
    L[lowthres] = 903.3 * cie[1, lowthres]
    # Calculate the a
    a = 500*(sugar_func(cie[0, ...]) - sugar_func(cie[1, ...]));
    # Calculate the b
    b = 200*(sugar_func(cie[1, ...]) - sugar_func(cie[2, ...]));

    #  Values lie between -128 < b <= 127, -128 < a <= 127, 0 <= L <= 100 
    Lab = np.stack([L, a, b], axis=0)
    return np.transpose(Lab, (1, 2, 0))

def cielab2cielch(lab_img):
    """Convert an image from CIEL*A*B* space to CIELCh space."""
    lch = np.zeros_like(lab_img)
    lch[..., 0] = lab_img[..., 0]
    lch[..., 1] = np.sqrt(np.square(lab_img[..., 1])+np.square(lab_img[..., 2]))
    lch[..., 2] = np.arctan2(lab_img[..., 2], lab_img[..., 1])
    return lch

