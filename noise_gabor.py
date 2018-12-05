# Implemntation modified from https://github.com/frankhjwx/Gabor_noise
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from math import cos, exp, floor, pi, sin
from random import randint, random

# Anisotropic Gabor noise
def gabor_noise(size, point_num, g_var, h_freq, h_omega, grid_size):
    
    # Gaussian kernel
    gauss_kern = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            x = (i-size / 2) / size
            y = (j-size / 2) / size
            gauss_kern[i][j] = exp(-pi * ((g_var * size)**2) * (x * x + y * y))
                
    # Harmonic kernel
    harmo_kern = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            x = (j - size / 2) / size
            y = (i - size / 2) / size
            value = cos(2*pi * (h_freq * size) * (x * cos(h_omega) + y * sin(h_omega)))
            harmo_kern[i][j] = value
    
    # Gabor kernel
    gabor_kern = np.multiply(gauss_kern, harmo_kern)
    
    # Sparse convolution noise
    sp_conv = np.zeros([size, size])
    dim = int(floor(size / 2 / grid_size))
    noise = []
    for i in range(-dim, dim + 1):
        for j in range(-dim, dim + 1):
            x = i * grid_size + size / 2 - grid_size / 2
            y = j * grid_size + size / 2 - grid_size / 2
            for _ in range(point_num):
                dx = randint(0, grid_size)
                dy = randint(0, grid_size)
                while not valid_position(size, x + dx, y + dy):
                    dx = randint(0, grid_size)
                    dy = randint(0, grid_size)
                weight = random() * 2 - 1
                sp_conv[int(x + dx)][int(y + dy)] = weight * 0.5 + 0.5

    return cv2.filter2D(sp_conv, -1, gabor_kern)

def valid_position(size, x, y):
    if x < 0 or x >= size: return False
    if y < 0 or y >= size: return False
    return True
