'''This part contains the code for the basic VAE (Variational Autoencoder) structure used in CVAE, TCVAE and FactorVAE.'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, GRU, Dense, Reshape, RepeatVector, Bidirectional, Activation,  Concatenate, Lambda, Add, BatchNormalization, Flatten
from tensorflow.keras import Model



def ReName (layer, name):
    return Lambda(lambda x: x, name=name)(layer)

## --------------------------------------------------    Models   ------------------------------------------------------------
## --------------------------------------------------  Model Parts  -------------------------------------------------------------
def EncoderBlock(x):
    x_input = x
    x = Bidirectional(GRU(25, return_sequences=True))(x)
    x = Bidirectional(GRU(25, return_sequences=True))(x)
    x = Bidirectional(GRU(25, return_sequences=False))(x)
    x = Add()([x, x_input])  # Residual Connection
    x = BatchNormalization()(x)
    return x

def DecoderBlock(x):
    x_input = x
    x = Bidirectional(GRU(25, return_sequences=True))(x)
    x = Bidirectional(GRU(25, return_sequences=True))(x)
    x = Bidirectional(GRU(int(x_input.shape[-1]//2), return_sequences=True))(x)
    x = Add()([x, x_input])  # Residual Connection
    x = BatchNormalization()(x)
    return x

def Reparameterization(Z_Mu, Z_Log_Sigma, ReparaStd=1., Reparam=False, num=''):
    # Reparameterization Trick for sampling from Guassian distribution
    Epsilon = tf.random.normal(shape=(tf.shape(Z_Mu)[0], Z_Mu.shape[1]), mean=0., stddev=ReparaStd)
        
    if Reparam==False:
        Epsilon = Epsilon * 0
    
    ZSamp = Z_Mu + tf.exp(0.5 * Z_Log_Sigma) * Epsilon
    ZSamp = ReName(ZSamp,'Zs'+str(num))

    return ZSamp    


## --------------------------------------------------   Encoder  -------------------------------------------------------------
def Encoder(SigDim, CondDim=None,  SlidingSize = 50, LatDim= 2, Type = 'Base',  ReparaStd = 0.1, Reparam = False):

    InpL = Input(shape=(SigDim,), name='Inp_Enc')
    InpFrame = tf.signal.frame(InpL, SlidingSize, SlidingSize)


    # Base VAE vs very deep VAE with hierachical latent vectors and residual connections
    if 'Base' in Type:
        Encoder = Bidirectional(GRU(25, return_sequences=True))(InpFrame)
        Encoder = Bidirectional(GRU(25, return_sequences=True))(Encoder)
        Encoder = Bidirectional(GRU(75, return_sequences=False))(Encoder)
    elif 'VDV' in Type:
        Encoder = EncoderBlock(InpFrame)
        Encoder = EncoderBlock(Encoder)
        Encoder = EncoderBlock(Encoder)
        Encoder = Flatten()(Encoder)
        
    
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

    
    if 'Base' in Type:
        Z_Mu = Dense(LatDim, activation='linear', name='Z_Mu')(Encoder)
        Z_Log_Sigma = Dense(LatDim, activation='softplus')(Encoder)
        Z_Log_Sigma = ReName(Z_Log_Sigma,'Z_Log_Sigma')
        Zs = Reparameterization(Z_Mu, Z_Log_Sigma, ReparaStd=ReparaStd, Reparam=Reparam)
        
    elif 'VDV' in Type:
        Z_Mus = []
        Z_Log_Sigmas = []
        Zs = []
        
        X = Dense(LatDim[0], activation='relu', name='Z_Mu0')(Encoder)
        for num, Dim in enumerate(LatDim):
            Z_Mu = Dense(Dim, activation='linear', name='Z_Mu'+str(num+1))(X)
            Z_Log_Sigma = Dense(Dim, activation='softplus')(X)
            Z_Log_Sigma = ReName(Z_Log_Sigma,'Z_Log_Sigma'+str(num+1))
            ZSamp = Reparameterization(Z_Mu, Z_Log_Sigma, ReparaStd=ReparaStd, Reparam=Reparam, num=num)
            
            Z_Mus.append(Z_Mu)
            Z_Log_Sigmas.append(Z_Log_Sigma)
            Zs.append(ZSamp)

            if num < len(LatDim) - 1:
                X = Dense(LatDim[-1], activation='relu')(ZSamp)
        
    return Model(ModelInp, Zs, name='EncModel') 

    

## --------------------------------------------------   Reconstructor  -------------------------------------------------------------
def Decoder(SigDim, CondDim=None, LatDim= 2, SlidingSize= 50):

    Nrepeat = SigDim//SlidingSize
    InpZ = Input(shape=(LatDim,), name='Inp_Z')
    
    # Conditional VAE vs other models
    if CondDim is not None:
        InpCond = Input(shape=(CondDim,), name='Inp_Cond')
        DecoderInp = Concatenate()([InpZ,InpCond])
        ModelInp = [InpZ, InpCond]
    else:
        DecoderInp = InpZ
        ModelInp = InpZ
        
    Decoder = Dense(LatDim, activation='relu')(DecoderInp)
    Decoder = Dense(LatDim, activation='relu')(Decoder)
    Decoder = Dense(SlidingSize, activation='relu')(Decoder)

    Decoder = RepeatVector(Nrepeat)(Decoder)  
    Decoder = DecoderBlock(Decoder)
    Decoder = DecoderBlock(Decoder)
    Decoder = DecoderBlock(Decoder)
    DecOut = Dense(SlidingSize,'sigmoid')(Decoder)
    DecOut = Reshape((SigDim,),name='Out')(DecOut)

        
    return Model(ModelInp, DecOut, name='ReconModel')



