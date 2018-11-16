import numpy as np
from noise import pnoise2

# Normalize vector
def normalize(vec):
    max = np.amax(vec)
    min  = np.amin(vec)
    return (vec - min) / (max - min)

# Generate Perlin noise with sine function mapping
# Assumes original image has shape (x_dim, y_dim, 3)
def perlin(x_dim, y_dim, period, octave, freq_sine):
    
    # Base Perlin noise
    noise = np.empty((x_dim, y_dim), dtype = np.float32)
    for x in range(x_dim):
        for y in range(y_dim):
            noise[x][y] = pnoise2(x / period, y / period, octaves = octave, lacunarity = 2)
    
    # Preprocessing and sine function color map
    noise = normalize(noise)
    noise = np.sin(noise * freq_sine * np.pi)
    noise = np.repeat(noise, 3)
    return noise.reshape(x_dim, y_dim, 3)

# Perturb original image and clip to maximum perturbation
def perturb(orig, max_norm, noise):
    noise = noise * 32
    noise = np.clip(noise, np.maximum(-orig, -max_norm), np.minimum(255 - orig, max_norm))
    return (orig + noise)