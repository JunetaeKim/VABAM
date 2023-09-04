'''This part contains the code for the basic VAE (Variational Autoencoder) structure used in CVAE, TCVAE and FactorVAE.'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, GRU, Dense, Reshape, RepeatVector, Bidirectional, Activation, GaussianNoise, Lambda, Concatenate
from tensorflow.keras import Model



def ReName (layer, name):
    return Lambda(lambda x: x, name=name)(layer)

## --------------------------------------------------    Models   ------------------------------------------------------------
## --------------------------------------------------   Encoder  -------------------------------------------------------------
def Encoder(SigDim, CondDim=None, SlidingSize = 50, LatDim= 2, Type = '',  ReparaStd = 0.1 , Reparam = False):

    InpL = Input(shape=(SigDim,), name='Inp_Enc')
    InpFrame = tf.signal.frame(InpL, SlidingSize, SlidingSize)

    #Encoder = Dense(50, activation='relu')(InpFrame)
    Encoder = Bidirectional(GRU(25, return_sequences=True))(InpFrame)
    Encoder = Bidirectional(GRU(25, return_sequences=True))(Encoder)
    Encoder = Bidirectional(GRU(75, return_sequences=False))(Encoder)
    
    # Conditional VAE vs other models
    if CondDim is not None:
        InpCond = Input(shape=(CondDim,), name='Inp_Cond')
        Encoder = Concatenate()([Encoder,InpCond])
        ModelInp = [InpL, InpCond]
    else:
        ModelInp = InpL
        
    Encoder = Dense(100, activation='relu')(Encoder)
    Encoder = Dense(70, activation='relu')(Encoder)
    Encoder = Dense(50, activation='relu')(Encoder)

    Z_Mu = Dense(LatDim, activation='linear', name='Z_Mu')(Encoder)
    Z_Log_Sigma = Dense(LatDim, activation='softplus')(Encoder)
    Z_Log_Sigma = ReName(Z_Log_Sigma,'Z_Log_Sigma'+Type)
    
    # Reparameterization Trick for sampling from Guassian distribution
    Epsilon = tf.random.normal(shape=(tf.shape(Z_Mu)[0], Z_Mu.shape[1]), mean=0., stddev=ReparaStd)
    
    if Reparam==False:
        Epsilon = Epsilon * 0

    Zs = Z_Mu + tf.exp(0.5 * Z_Log_Sigma) * Epsilon
    Zs = ReName(Zs,'Zs'+Type)

        
    return Model(ModelInp, Zs, name='EncModel') 

    

## --------------------------------------------------   Reconstructor  -------------------------------------------------------------
def Decoder(SigDim, CondDim=None, LatDim= 2, SlidingSize= 50):

    InpZ = Input(shape=(LatDim,), name='Inp_Z')
    
    # Conditional VAE vs other models
    if CondDim is not None:
        InpCond = Input(shape=(CondDim,), name='Inp_Cond')
        DecoderInp = Concatenate()([InpZ,InpCond])
        ModelInp = [InpZ, InpCond]
    else:
        DecoderInp = InpZ
        ModelInp = InpZ
        
    Decoder = Dense(50, activation='relu')(DecoderInp)
    Decoder = Dense(50, activation='relu')(Decoder)
    Decoder = Dense(50, activation='relu')(Decoder)
    Decoder = Dense(75, activation='relu')(Decoder)
    Decoder = RepeatVector((SigDim//SlidingSize) )(Decoder)
    Decoder = Bidirectional(GRU(25, return_sequences=True))(Decoder)
    Decoder = Bidirectional(GRU(25, return_sequences=True))(Decoder)
    Decoder = Bidirectional(GRU(25, return_sequences=True))(Decoder)

    DecOut = Dense(SlidingSize,'sigmoid')(Decoder)
    DecOut = Reshape((SigDim,),name='Out')(DecOut)

        
    return Model(ModelInp, DecOut, name='ReconModel')



