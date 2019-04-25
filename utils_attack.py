from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

'''
Color noise

noise           has dimension 2 or 3, pixel range [0, 255]
color            is [a, b, c] where a, b, c are from {-1, 0, 1}
'''
def colorize(noise, color = [1, 1, 1]):
    if noise.ndim == 2: # expand to include color channels
        noise = np.expand_dims(noise, 2)
    return (noise - 0.5) * color * 2 # output pixel range [-1, 1]

'''
Perturb image and clip to maximum perturbation norm

img              image with pixel range [0, 1]
noise           noise with pixel range [-1, 1]
norm           L-infinity norm constraint
'''
def perturb(img, noise, norm):
    noise = np.sign((noise - 0.5) * 2) * norm
    noise = np.clip(noise, np.maximum(-img, -norm), np.minimum(255 - img, norm))
    return (img + noise)
