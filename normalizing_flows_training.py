import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
import seaborn as sns

from flow import *

z = np.load('data/latent_space.npy',mmap_mode = 'c')
z_2 = np.load('data/latent_space_2.npy',mmap_mode = 'c')

z = np.concatenate((z,z_2), axis = 0)

hidden_dim = [256,256]
layers =8
bijectors = []
for i in range(0, layers):
    made = make_network(32, hidden_dim,2)
    bijectors.append(MAF(made))
    bijectors.append(tfb.Permute(permutation=[31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0])) 
    
bijectors = tfb.Chain(bijectors=list(reversed(bijectors[:-1])))

distribution = tfd.TransformedDistribution(
    distribution=tfd.Normal(loc=0., scale=1.),
    bijector=bijectors,
    event_shape=[32]
)

x_ = tfkl.Input(shape=(32,), dtype=tf.float32)
log_prob_ = distribution.log_prob(x_)
model = tfk.Model(x_, log_prob_)

model.compile(optimizer=tf.optimizers.Adam(), loss=lambda _, log_prob: -log_prob)

model.summary()

# loading_path = 'nflow_weights/'
# latest = tf.train.latest_checkpoint(loading_path)
# model.load_weights(latest)

_ = model.fit(x=z,
              y=np.zeros((z.shape[0], 0), dtype=np.float32),
              batch_size= z.shape[0],
              epochs=30000,
              steps_per_epoch=1,
              verbose=1, 
              shuffle=False)

saving_path = 'nflow_weights/'
model.save_weights(saving_path+'test')