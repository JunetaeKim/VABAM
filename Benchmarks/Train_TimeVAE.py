'''
Notice
The original code for GP-VAE can be found in the author's repository. https://github.com/abudesai/timeVAE
'''

import os
import sys
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from Libs.TIMEVAE.vae_dense_model import VariationalAutoencoderDense as VAE_Dense
from Libs.TIMEVAE.vae_conv_model import VariationalAutoencoderConv as VAE_Conv
from Libs.TIMEVAE.vae_conv_I_model import VariationalAutoencoderConvInterpretable as TimeVAE
import Libs.TIMEVAE.utils
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr
from argparse import ArgumentParser


## GPU selection
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
    LatDim = 10

    ### Other parameters
    Patience = 300
    BatchSize = 10000
    NEpoch = 1000


    # Create the parser
    parser = ArgumentParser()
    parser.add_argument('--SigType', type=str, required=True, help='Types of signals to train on.: ART, PLETH, II')

    args = parser.parse_args() # Parse the arguments
    SigType = args.SigType
    #assert SigType in ['ART', 'PLETH', 'II', 'filtII'], "Value should be either ART, PLETH, II."

    
    file_pref = 'TimeVae_Lat'+str(LatDim)+'_' 
    Outdir = './Results/'
    if not os.path.exists(Outdir):
        os.mkdir(Outdir)
        
    
    #### -----------------------------------------------------   Data load and processing   -------------------------------------------------------------------------    
    TrData = np.load('../Data/ProcessedData/Tr'+str(SigType)+'.npy').astype('float32')
    ValData = np.load('../Data/ProcessedData/Val'+str(SigType)+'.npy').astype('float32')


    TrDataFrame = tf.signal.frame(TrData, 50, 50).numpy()
    ValDataFrame = tf.signal.frame(ValData, 50, 50).numpy()

  
    
    #### -----------------------------------------------------   Model   -------------------------------------------------------------------------    
    vae = TimeVAE( seq_len=TrDataFrame.shape[1],  feat_dim = TrDataFrame.shape[2], latent_dim = LatDim, hidden_layer_sizes=[50, 100, 200], reconstruction_wt = 3.0, use_residual_conn = True)   

    vae.compile(optimizer=Adam())

    print(vae.summary())
           
    
    #### -----------------------------------------------------   Model Fit   -----------------------------------------------------------------------    
    # Load weights
    if os.path.isfile(Outdir + file_pref+'decoder_wts.h5'):
        vae = TimeVAE.load(Outdir, file_pref)


    early_stop_loss = 'val_loss'
    early_stop_callback = EarlyStopping(monitor=early_stop_loss, min_delta = 1e-1, patience=Patience) 
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

    vae.fit( TrDataFrame, validation_data=(ValDataFrame,None),  batch_size = BatchSize, epochs=NEpoch,  shuffle = True,  callbacks=[early_stop_callback, reduceLR, ],   verbose = 1)

 
    # save model 
    vae.save(Outdir, file_pref)