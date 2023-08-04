'''
Notice
The original author's cost function has been modified in this script: instead of maximizing p(x|z), the minimization of MSE (Mean Squared Error) is used.
In the original author's paper, the study was conducted with relatively short timesteps, so latent variables zs were generated for all timesteps. 
However, in the case of this study, the raw dimensional setting has 1000 timesteps (10 seconds with a sample rate of 100), which becomes an overly burdensome setting. 
Therefore, latent variables zs were generated for every second instead. In other words, z values were generated for a total of 10 seconds across the timesteps.
The original code for HI-VAE can be found in the author's repository. https://github.com/ratschlab/GP-VAE
'''

import time
from datetime import datetime
import sys
import os
from argparse import ArgumentParser

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
from Libs.GPVAE.models_varia_mse import *


## GPU selection
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

# TensorFlow wizardry
config = tf.compat.v1.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 1.0
# Create a session with the above options specified.
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))     


if __name__ == "__main__":

    #### -----------------------------------------------------   Experiment setting   -------------------------------------------------------------------------    
    ### Model related parameters
    num_steps = 0 # 'Number of training steps: If non-zero it overwrites num_epochs'
    num_epochs = 2000
    batch_size = 4000
    print_interval = 0
    TrRate = 0.8
    LatDim = 10
    learning_rate = 0.001
    gradient_clip = 1e4
    debug= True
    
    # Create the parser
    parser = ArgumentParser()
    parser.add_argument('--SigType', type=str, required=True, help='Types of signals to train on.: ART, PLETH, II')
    
    args = parser.parse_args() # Parse the arguments
    SigType = args.SigType
    #assert SigType in ['ART', 'PLETH', 'II', 'filtII'], "Value should be either ART, PLETH, II."
    
    
    Outdir = './Results/'
    if not os.path.exists(Outdir):
        os.mkdir(Outdir)
        
    
    #### -----------------------------------------------------   Data load and processing   -------------------------------------------------------------------------    
    TrData = np.load('../Data/ProcessedData/Tr'+str(SigType)+'.npy').astype('float32')
    ValData = np.load('../Data/ProcessedData/Val'+str(SigType)+'.npy').astype('float32')
  

    TrDataFrame = tf.signal.frame(TrData, 50, 50).numpy()
    ValDataFrame = tf.signal.frame(ValData, 50, 50).numpy()
    # 0 is missing, 1 is observed
    m_train_miss = np.zeros_like(TrDataFrame)
    m_val_miss = np.zeros_like(ValDataFrame)

    tf_x_train_miss = tf.data.Dataset.from_tensor_slices((TrDataFrame, m_train_miss)).shuffle(len(TrDataFrame)).batch(batch_size).repeat()
    tf_x_val_miss = tf.data.Dataset.from_tensor_slices((ValDataFrame, m_val_miss)).batch(batch_size).repeat()
    tf_x_val_miss = tf.compat.v1.data.make_one_shot_iterator(tf_x_val_miss)

    data_dim = TrDataFrame.shape[-1]
    time_length = TrDataFrame.shape[1]

    if num_steps == 0:
        num_steps = num_epochs * len(TrDataFrame) // batch_size
    else:
        num_steps = num_steps

    if print_interval == 0:
        print_interval = num_steps // num_epochs

  
    
    
    #### -----------------------------------------------------   Model   -------------------------------------------------------------------------    
    model = HI_VAE(latent_dim=LatDim, 
                   data_dim=data_dim, 
                   time_length=time_length,
                   encoder_sizes=[50, 40, 30], encoder=JointEncoderGRU,
                   decoder_sizes=[30,40,50], decoder=GaussianDecoder,
                   M=1, K=1,  )
    
    _ = tf.compat.v1.train.get_or_create_global_step()
    trainable_vars = model.get_trainable_vars()
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)

    print("Encoder: ", model.encoder.net.summary())
    print("Decoder: ", model.decoder.net.summary())
       
    
    #### -----------------------------------------------------   Model Fit   -----------------------------------------------------------------------    
    # Load weights
    if os.path.isfile(Outdir+SigType+'_HIVAE_Encoder.hdf5'):
        model.encoder.net.load_weights(Outdir+SigType+'_HIVAE_Encoder.hdf5')
    if os.path.isfile(Outdir+SigType+'_HIVAE_Decoder.hdf5'):
        model.decoder.net.load_weights(Outdir+SigType+'_HIVAE_Decoder.hdf5')


    losses_train = []
    losses_val = []

    t0 = time.time()
    val_loss_check = np.inf
    
    for i, (x_seq, m_seq) in enumerate(tf_x_train_miss.take(num_steps)):
        with tf.GradientTape() as tape:
            tape.watch(trainable_vars)
            loss = model.compute_loss(x_seq, m_mask=m_seq)
            losses_train.append(loss.numpy())
        grads = tape.gradient(loss, trainable_vars)
        grads = [np.nan_to_num(grad) for grad in grads]
        grads, global_norm = tf.clip_by_global_norm(grads, gradient_clip)
        optimizer.apply_gradients(zip(grads, trainable_vars),
                                  global_step=tf.compat.v1.train.get_or_create_global_step())


        
        # Print intermediate results
        if i % print_interval == 0:
            print("================================================")
            print("Learning rate: {} | Global gradient norm: {:.2f}".format(optimizer._lr, global_norm))
            print("Step {}) Time = {:2f}".format(i, time.time() - t0))
            loss, mse, kl = model.compute_loss(x_seq, m_mask=m_seq, return_parts=True)
            print("Train loss = {:.5f} | MSE = {:.5f} | KL = {:.5f}".format(loss, mse, kl))


            tf.summary.scalar("loss_train", loss, step=i)
            tf.summary.scalar("kl_train", kl, step=i)
            tf.summary.scalar("mse_train", mse, step=i)

            # Validation loss
            x_val_batch, m_val_batch = tf_x_val_miss.get_next()
            val_loss, val_mse, val_kl = model.compute_loss(x_val_batch, m_mask=m_val_batch, return_parts=True)
            losses_val.append(val_loss.numpy())
            print("Validation loss = {:.5f} | MSE = {:.5f} | KL = {:.5f}".format(val_loss, val_mse, val_kl))

            tf.summary.scalar("loss_val", val_loss, step=i)
            tf.summary.scalar("kl_val", val_kl, step=i)
            tf.summary.scalar("mse_val", val_mse, step=i)
            x_hat = model.decode(model.encode(x_seq).sample()).mean()

            if val_loss_check > val_loss:
                print(val_loss_check, val_loss)
                val_loss_check = val_loss
                model.encoder.net.save_weights(Outdir+SigType+'_HIVAE_Encoder.hdf5')
                model.decoder.net.save_weights(Outdir+SigType+'_HIVAE_Decoder.hdf5')



            # Update learning rate (used only for physionet with decay=0.5)
            if i > 0 and i % (10*print_interval) == 0:
                optimizer._lr = max(0.99 * optimizer._lr, 0.1 * learning_rate)
            t0 = time.time()

