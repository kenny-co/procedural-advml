from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from noise import pnoise2
from utils_attack import colorize
from utils_noise import gaborN_rand, gaborN_uni
from utils_noise import normalize, perlin

## Helper functions

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
def get_noise_f(size, noise_f):
    
    # Gabor noise - random spread
    if noise_f == 'gabor_rand':
        pass
    
    # Gabor noise - uniform spread
    if noise_f == 'gabor_uni':
        pass
        
    # Perlin noise
    if noise_f == 'perlin':
        def noise_func(params):
            freq, freq_sine, octave = params
            noise = perlin(size, 1 / freq, int(octave), freq_sine)
            return colorize(noise)
        
        # Parameter boundaries for Bayesian optimization
        bounds = [{'name' : 'freq', 'type' : 'continuous', 'domain' : (1 / 160, 1 / 20), 'dimensionality' : 1},
                  {'name' : 'freq_sine', 'type' : 'continuous', 'domain' : (4, 32), 'dimensionality' : 1},
                  {'name' : 'octave'  , 'type' : 'discrete'  , 'domain' : (1, 2, 3, 4), 'dimensionality' : 1}]
        
    return noise_func, bounds
