from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from noise_gabor import gabor_noise
from noise import pnoise2
from opensimplex import OpenSimplex

## Helper functions

# Normalize vector
def normalize(vec):
    vmax = np.amax(vec)
    vmin  = np.amin(vec)
    return (vec - vmin) / (vmax - vmin)

# Perturb original image and clip to maximum perturbation
def perturb(orig, max_norm, noise):
    noise = np.sign(noise) * max_norm
    noise = np.clip(noise, np.maximum(-orig, -max_norm), np.minimum(255 - orig, max_norm))
    return (orig + noise)

## Noise generating functions
# Assumes original image has shape (dim_x, dim_y, 3)

# Generate anisotropic Gabor noise
# Assumes square image (dim_x = dim_y)
def gabor_ani(dim_x, point_num, g_var, h_freq, h_omega, freq_sine, grid_size = 23):
    noise = gabor_noise(size = dim_x, point_num = point_num, g_var = g_var, h_freq = h_freq, h_omega = h_omega, grid_size = grid_size)
    noise = normalize(noise)
    noise = np.sin(noise * freq_sine * np.pi)
    noise = np.repeat(noise, 3)
    return noise.reshape(dim_x, dim_x, 3)

# Generate isotropic Gabor noise
# Assumes square image (dim_x = dim_y)
def gabor_iso(dim_x, point_num, g_var, h_freq, freq_sine, comp = 5, grid_size = 23):
    
    # Combine anisotropic signal to produce pseudo-isotropic signal
    noise = gabor_noise(size=dim_x, point_num = point_num, g_var = g_var, h_freq = h_freq, h_omega = np.pi * (comp - 1) / comp, grid_size = grid_size)
    for i in range(comp):
        noise += gabor_noise(size=dim_x, point_num = point_num, g_var = g_var, h_freq = h_freq, h_omega = np.pi * i / comp, grid_size = grid_size)
    
    noise = normalize(noise)
    noise = np.sin(noise * freq_sine * np.pi)
    noise = np.repeat(noise, 3)
    return noise.reshape(dim_x, dim_x, 3)

# Generate OpenSimplex noise with sine function mapping
def osimplex(dim_x, dim_y, period_x, period_y, freq_sine):
    tmp = OpenSimplex()
    
    # OpenSimplex noise
    noise = np.empty((dim_x, dim_y), dtype = np.float32)
    for x in range(dim_x):
        for y in range(dim_y):
            noise[x][y] = tmp.noise2d(x / period_x, y / period_y)
            
    # Preprocessing and sine function color map
    noise = normalize(noise)
    noise = np.sin(noise * freq_sine * np.pi)
    noise = np.repeat(noise, 3)
    return noise.reshape(dim_x, dim_y, 3)

# Generate Perlin noise with sine function mapping
def perlin(dim_x, dim_y, period_x, period_y, octave, freq_sine, lacunarity = 2):
    
    # Perlin noise
    noise = np.empty((dim_x, dim_y), dtype = np.float32)
    for x in range(dim_x):
        for y in range(dim_y):
            noise[x][y] = pnoise2(x / period_x, y / period_y, octaves = octave, lacunarity = lacunarity)
            
    # Preprocessing and sine function color map
    noise = normalize(noise)
    noise = np.sin(noise * freq_sine * np.pi)
    noise = np.repeat(noise, 3)
    return noise.reshape(dim_x, dim_y, 3)
