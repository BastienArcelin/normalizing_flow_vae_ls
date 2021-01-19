# Import packages
import numpy as np
import os
import tensorflow as tf
import pathlib
from pathlib import Path

import model, vae_functions



def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

############## NORMALIZATION OF IMAGES
# Value used for normalization
beta = 2.5

def norm(x, bands, path , channel_last=False, inplace=True):
    '''
    Return image x normalized

    Parameters:
    -----------
    x: image to normalize
    bands: filter number
    path: path to the normalization constants
    channel_last: is the channels (filters) in last in the array shape
    inplace: boolean: change the value of array itself
    '''
    while bands[0]>9:
        bands = np.array(bands)-10
    full_path = pathlib.PurePath(path)
    isolated_or_blended = full_path.parts[6][0:len(full_path.parts[6])-9]

    test_dir = str(Path(path).parents[0])+'/norm_cst/'
    I = np.load(test_dir+'galaxies_isolated_20191024_0_I_norm.npy', mmap_mode = 'c')
    if not inplace:
        y = np.copy(x)
    else:
        y = x
    if channel_last:
        assert y.shape[-1] == len(bands)
        for i in range (len(y)):
            for ib, b in enumerate(bands):
                y[i,:,:,ib] = np.tanh(np.arcsinh(y[i,:,:,ib]/(I[b]/beta)))
    else:
        assert y.shape[1] == len(bands)
        for i in range (len(y)):
            for ib, b in enumerate(bands):
                y[i,ib] = np.tanh(np.arcsinh(y[i,ib]/(I[b]/beta)))
    return y

def denorm(x, bands ,path , channel_last=False, inplace=True):
    '''
    Return image x denormalized

    Parameters:
    ----------
    x: image to denormalize
    bands: filter number
    path: path to the normalization constants
    channel_last: is the channels (filters) in last in the array shape
    inplace: boolean: change the value of array itself
    '''
    while bands[0]>9:
        bands = np.array(bands)-10
    full_path = pathlib.PurePath(path)
    isolated_or_blended = full_path.parts[6][0:len(full_path.parts[6])-9]
    #print(isolated_or_blended)
    test_dir = str(Path(path).parents[0])+'/norm_cst/'
    I = np.load(test_dir+'galaxies_'+isolated_or_blended+'_20191024_0_I_norm.npy', mmap_mode = 'c')#I = np.concatenate([I_euclid,n_years*I_lsst])
    if not inplace:
        y = np.copy(x)
    else:
        y = x
    if channel_last:
        print(y.shape)
        assert y.shape[-1] == len(bands)
        for i in range (len(y)):
            for ib, b in enumerate(bands):
                y[i,:,:,ib] = np.sinh(np.arctanh(y[i,:,:,ib]))*(I[b]/beta)
    else:
        assert y.shape[1] == len(bands)
        for i in range (len(y)):
            for ib, b in enumerate(bands):
                y[i,ib] = np.sinh(np.arctanh(x[i,ib]))*(I[b]/beta)
    return y



def load_vae_full(path, nb_of_bands, folder=False):
    """
    Return the loaded VAE, outputs for plotting evlution of training, the encoder, the decoder and the Kullback-Leibler divergence 

    Parameters:
    ----------
    path: path to saved weights
    nb_of_bands: number of filters to use
    folder: boolean, change the loading function
    """        
    latent_dim = 32
    
    # Build the encoder and decoder
    encoder, decoder = model.vae_model(latent_dim, nb_of_bands)

    # Build the model
    vae_loaded, vae_utils,  Dkl = vae_functions.build_vanilla_vae(encoder, decoder, full_cov=False, coeff_KL = 0)

    if folder == False: 
        vae_loaded.load_weights(path)
    else:
        print(path)
        latest = tf.train.latest_checkpoint(path)
        vae_loaded.load_weights(latest)

    return vae_loaded, vae_utils, encoder, decoder, Dkl

