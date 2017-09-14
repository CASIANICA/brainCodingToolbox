# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np

def sugar_func(t):
    """Util function for rgb2cielab"""
    if (t > 0.008856):
        return np.power(t, 1/3.0);
    else:
        return 7.787 * t + 16 / 116.0;

def rgb2cielab(rgb_image):
    """Convert an image in RGB space to CIEL*A*B* space."""
    #Conversion Matrix
    matrix = [[0.412453, 0.357580, 0.180423],
              [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]]

    # RGB values lie between 0 to 1.0
    rgb = [1.0, 0, 0] # RGB

    cie = np.dot(matrix, rgb);

    cie[0] = cie[0] /0.950456;
    cie[2] = cie[2] /1.088754; 

    # Calculate the L
    L = 116 * np.power(cie[1], 1/3.0) - 16.0 if cie[1] > 0.008856 else 903.3 * cie[1];

    # Calculate the a 
    a = 500*(sugar_func(cie[0]) - sugar_func(cie[1]));

    # Calculate the b
    b = 200*(sugar_func(cie[1]) - sugar_func(cie[2]));

    #  Values lie between -128 < b <= 127, -128 < a <= 127, 0 <= L <= 100 
    Lab = [b , a, L]; 

    # OpenCV Format
    L = L * 255 / 100;
    a = a + 128;
    b = b + 128;
    Lab_OpenCV = [b , a, L];

