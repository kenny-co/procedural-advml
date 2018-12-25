from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from noise import pnoise2
from opensimplex import OpenSimplex


## Helper functions

# Normalize vector to range (0, 1)
def normalize(vec):
    vmax = np.amax(vec)
    vmin  = np.amin(vec)
    return (vec - vmin) / (vmax - vmin)

# Perturb original image and clip to maximum perturbation
def perturb(orig, max_norm, noise):
    orig = orig.reshape((299, 299, 3))
    noise = noise * 16
    noise = np.clip(noise, np.maximum(-orig, -max_norm), np.minimum(255 - orig, max_norm))
    return orig + noise


## Load predict function of model
def load_predict(model_name):
    
    # Inception v3
    if model_name == 'IncV3':
        from keras.applications.inception_v3 import InceptionV3
        from keras.applications.inception_v3 import decode_predictions, preprocess_input
        model = InceptionV3(weights = 'imagenet')
        
        def predict_prob(vec):
            img  = vec.reshape((1, 299, 299, 3)).astype(np.float)
            pred = model.predict(preprocess_input(img))
            return pred[0], decode_predictions(pred, top = 6)[0]
        
    return predict_prob


## Noise generating functions with sine function mapping

# Assumes original image has shape (dim, dim, 3)
# Includes bounds for Bayesian optimization
def get_noise_f(dim, noise_f):
    
    # OpenSimplex noise
    if noise_f == 'osimplex':
        def noise_func(params):
            period_x, period_y, freq_sin = params
            tmp = OpenSimplex()
            
            # Base OpenSimplex noise
            noise = np.empty((dim, dim), dtype = np.float32)
            for x in range(dim):
                for y in range(dim):
                    noise[x][y] = tmp.noise2d(x * freq, y * freq)
                    
            # Preprocessing and sine function color map
            noise = normalize(noise)
            noise = np.sin(noise * freq_sin * np.pi)
            noise = np.repeat(noise, 3)
            return noise.reshape(dim, dim, 3)
        
        # Parameter boundaries for Bayesian optimization
        bounds = [{'name' : 'freq', 'type' : 'continuous', 'domain' : (1 / 160, 1 / 20)    , 'dimensionality' : 1},
                  {'name' : 'freq_sin', 'type' : 'continuous', 'domain' : (4, 32)     , 'dimensionality' : 1}]
        
    # Perlin noise
    elif noise_f == 'perlin':
        def noise_func(params):
            freq, freq_sin, octave = params
            octave = int(octave)
            
            # Base Perlin noise
            noise = np.empty((dim, dim), dtype = np.float32)
            for x in range(dim):
                for y in range(dim):
                    noise[x][y] = pnoise2(x * freq, y * freq, octaves = octave)
                    
            # Preprocessing and sine function color map
            noise = normalize(noise)
            noise = np.sin(noise * freq_sin * np.pi)
            noise = np.repeat(noise, 3)
            return noise.reshape(dim, dim, 3)
        
        # Parameter boundaries for Bayesian optimization
        bounds = [{'name' : 'freq', 'type' : 'continuous', 'domain' : (1 / 160, 1 / 20)    , 'dimensionality' : 1},
                  {'name' : 'freq_sin', 'type' : 'continuous', 'domain' : (4, 32)     , 'dimensionality' : 1},
                  {'name' : 'octave'  , 'type' : 'discrete'  , 'domain' : (1, 2, 3, 4), 'dimensionality' : 1}]
        
    return noise_func, bounds
