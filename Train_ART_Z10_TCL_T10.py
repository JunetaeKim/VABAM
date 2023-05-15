import os
import sys
import numpy as np
import pandas as pd


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, GRU, Dense, Masking, Reshape, Flatten, RepeatVector, TimeDistributed, Bidirectional, Activation, GaussianNoise, Lambda, LSTM
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from Models.BioSigBandVAE_MultiM_Exp_deep_TCL_T10 import *
from Utilities.Utilities import *


## GPU selection
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
    MaskingRate = 0.0
    NoiseStd = 0.002
    MaskStd = 0.1
    ReparaStd = 0.1
    Capacity_Z = 0.1
    Capacity_Fc = 0.1
    FcLimit = 0.05
    DecayH = 0. 
    DecayL = 0.
    SlidingSize = 100
    
    ### Other parameters
    Patience = 300
    TrRate = 0.8
    
    SavePath = './Results/'
    ModelName = 'ART_Z'+str(LatDim)+'_TCL_T10.hdf5'
    
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
        
    ### Model checkpoint
    ModelSaveName = SavePath+ModelName
    

    
    #### -----------------------------------------------------   Data load and processing   -------------------------------------------------------------------------    
    TrData = np.load('./Data/ProcessedData/TrART.npy')
    ValData = np.load('./Data/ProcessedData/ValART.npy')
    SigDim = TrData.shape[1]
    
    
    #### -----------------------------------------------------   Model   -------------------------------------------------------------------------    
    EncModel = Encoder(SigDim=SigDim, SlidingSize=SlidingSize, LatDim= LatDim, Type = '', MaskingRate = MaskingRate, NoiseStd = NoiseStd, MaskStd = MaskStd, ReparaStd = ReparaStd, Reparam=True, FcLimit=FcLimit)
    FeatExtModel = FeatExtractor(SigDim=SigDim, DecayH=DecayH, DecayL=DecayL)
    FeatGenModel = FeatGenerator(SigDim=SigDim, SlidingSize=SlidingSize, LatDim= LatDim)
    ReconModel = Reconstructor(SigDim=SigDim, SlidingSize=SlidingSize, FeatDim=400)

    ## Model core parts
    EncInp =EncModel.input
    InpZ = EncModel.output[2]
    InpFCCommon = EncModel.output[1][:, :2]
    InpFCEach = EncModel.output[1][:, 2:]

    ## Each output of each model
    FeatExtOut = FeatExtModel(EncModel.output[:2])
    FeatGenOut = FeatGenModel([InpFCCommon, InpFCEach, InpZ])
    ReconExtOut = ReconModel(FeatExtOut)
    ReconGenOut = ReconModel(FeatGenOut)

    ### Define the total model
    SigBandRepModel = Model(EncInp, [FeatGenOut, ReconExtOut, ReconGenOut])

    ### Weight controller; Apply beta and capacity 
    Beta_Z = Lossweight(name='Beta_Z', InitVal=0.05)(FeatGenOut)
    Beta_Fc = Lossweight(name='Beta_Fc', InitVal=0.05)(FeatGenOut)
    Beta_TC = Lossweight(name='Beta_TC', InitVal=0.05)(FeatGenOut)
    Beta_Rec_ext = Lossweight(name='Beta_Rec_ext', InitVal=500.)(FeatGenOut)
    Beta_Rec_gen = Lossweight(name='Beta_Rec_gen', InitVal=500.)(FeatGenOut)
    Beta_Feat = Lossweight(name='Beta_Feat', InitVal=500.)(FeatGenOut)

    ### Adding the RecLoss; 
    MSE = tf.keras.losses.MeanSquaredError()
    
    ReconOut_ext = Beta_Rec_ext * MSE(ReconExtOut, EncInp)
    SigBandRepModel.add_loss(ReconOut_ext)
    SigBandRepModel.add_metric(ReconOut_ext, 'ReconOut_ext')
    
    #ReconOut_gen = Beta_Rec_gen * MSE(ReconGenOut, EncInp)
    #SigBandRepModel.add_loss(ReconOut_gen)
    #SigBandRepModel.add_metric(ReconOut_gen, 'ReconOut_gen')
    

    ### Adding the FeatRecLoss; It allows connection between the extractor and generator
    FeatRecLoss= Beta_Feat * MSE(tf.concat(FeatGenOut, axis=-1), tf.concat(FeatExtOut, axis=-1))
    SigBandRepModel.add_loss(FeatRecLoss)
    SigBandRepModel.add_metric(FeatRecLoss, 'FeatRecLoss')

    ### KL Divergence for p(Z) vs q(Z)
    Z_Mu, Z_Log_Sigma, Zs = SigBandRepModel.get_layer('Z_Mu').output, SigBandRepModel.get_layer('Z_Log_Sigma').output, SigBandRepModel.get_layer('Zs').output
    kl_Loss_Z = 0.5 * tf.reduce_sum( Z_Mu**2  +  tf.exp(Z_Log_Sigma)- Z_Log_Sigma-1, axis=1)    
    kl_Loss_Z = tf.reduce_mean(kl_Loss_Z )
    kl_Loss_Z = Beta_Z * tf.abs(kl_Loss_Z - Capacity_Z)

    ### KL Divergence for p(FCs) vs q(FCs)
    BernP = 0.5 # hyperparameter
    FCs = SigBandRepModel.get_layer('FC_Mu').output 
    kl_Loss_FC = tf.math.log(FCs) - tf.math.log(BernP) + tf.math.log(1-FCs) - tf.math.log(1-BernP) 
    kl_Loss_FC = tf.reduce_mean(-kl_Loss_FC )
    kl_Loss_FC = Beta_Fc * tf.abs(kl_Loss_FC - Capacity_Fc)

    
    ### KL Divergence for q(Z) vs q(Z)_Prod
    LogProb_QZ = LogNormalDensity(Zs[:, None], Z_Mu[None], Z_Log_Sigma[None])
    Log_QZ_Prod = tf.reduce_sum( tf.reduce_logsumexp(LogProb_QZ, axis=1, keepdims=False),   axis=1,  keepdims=False)
    Log_QZ = tf.reduce_logsumexp(tf.reduce_sum(LogProb_QZ, axis=2, keepdims=False),   axis=1,   keepdims=False)
    kl_Loss_TC = -tf.reduce_mean(Log_QZ - Log_QZ_Prod)
    kl_Loss_TC = Beta_TC * kl_Loss_TC
    
    
    SigBandRepModel.add_loss(kl_Loss_Z )
    SigBandRepModel.add_metric(kl_Loss_Z , 'kl_Loss_Z')

    SigBandRepModel.add_loss(kl_Loss_FC )
    SigBandRepModel.add_metric(kl_Loss_FC , 'kl_Loss_FC')

    SigBandRepModel.add_loss(kl_Loss_TC )
    SigBandRepModel.add_metric(kl_Loss_TC , 'kl_Loss_TC')

    
    ## Model Compile
    SigBandRepModel.compile(optimizer='adam') 


    ### Loss and KLD_Beta controller
    #KLD_Beta_Z = KLAnneal(TargetLossName=['val_FeatRecLoss', 'val_RecLoss'], Threshold=0.001, BetaName='Beta_Z',  MaxBeta=0.1 , MinBeta=0.1, AnnealEpoch=1, UnderLimit=1e-7, verbose=2)
    #KLD_Beta_Fc = KLAnneal(TargetLossName=['val_FeatRecLoss', 'val_RecLoss'], Threshold=0.001, BetaName='Beta_Fc',  MaxBeta=0.005 , MinBeta=0.005, AnnealEpoch=1, UnderLimit=1e-7, verbose=1)

    RelLossDic = { 'val_ReconOut_ext':'Beta_Rec_ext', 'val_FeatRecLoss':'Beta_Feat', 'val_kl_Loss_Z':'Beta_Z', 'val_kl_Loss_FC':'Beta_Fc', 'val_kl_Loss_TC':'Beta_TC'}
    ScalingDic = { 'val_ReconOut_ext':100., 'val_FeatRecLoss':200., 'val_kl_Loss_Z':0.1, 'val_kl_Loss_FC':0.1, 'val_kl_Loss_TC':0.1}
    MinLimit = {'Beta_Rec_ext':1., 'Beta_Feat':1., 'Beta_Z':0.01, 'Beta_Fc':0.01, 'Beta_TC':0.01}
    MaxLimit = {'Beta_Rec_ext':500., 'Beta_Feat':500., 'Beta_Z':0.25, 'Beta_Fc':0.25, 'Beta_TC':0.25}
    RelLoss = RelLossWeight(BetaList=RelLossDic, LossScaling= ScalingDic, MinLimit= MinLimit, MaxLimit = MaxLimit, ToSaveLoss=['val_FeatRecLoss', 'val_ReconOut_ext'] , SaveWay='max' , SavePath = ModelSaveName, CheckPoint=200)
    
    
       
    
    # Model Training
    #SigBandRepModel.load_weights(ModelSaveName)
    SigBandRepModel.fit(TrData, batch_size=3000, epochs=2000, shuffle=True, validation_data =(ValData, None) , callbacks=[  RelLoss]) 


