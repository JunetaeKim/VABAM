import os
import sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, GRU, Dense, Masking, Reshape, Flatten, RepeatVector, TimeDistributed, Bidirectional, Activation, GaussianNoise, Lambda, LSTM
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from Models.FPVAE_T20 import *
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

    
    # Create the parser
    parser = ArgumentParser()
    
    # Add Model related parameters
    parser.add_argument('--LatDim', type=int, required=True, help='The dimensionality of the latent variable z.')
    parser.add_argument('--SigType', type=str, required=True, help='Types of signals to train on.: ART, PLETH, II, filtII')
    
    parser.add_argument('--CompSize', type=int, required=False,default=500, help='Signal compression length (unit 0.01 second)')
    parser.add_argument('--MaskingRate', type=float, required=False, default=0.00, help='The sequence masking ratio refers to the proportion of the sequence that will be masked during training.')
    parser.add_argument('--NoiseStd', type=float, required=False, default=0.00, help='The standard deviation value for Gaussian noise generation across the entire signal.')
    parser.add_argument('--MaskStd', type=float, required=False, default=0.00, help='The standard deviation value for Gaussian noise generation applied to the masked signal.')
    parser.add_argument('--ReparaStd', type=float, required=False, default=0.1, help='The standard deviation value for Gaussian noise generation used in the reparametrization trick.')
    parser.add_argument('--Capacity_Z', type=float, required=False, default=0.1, help='The capacity value for controlling the Kullback-Leibler divergence (KLD) of Z.')
    parser.add_argument('--Capacity_Fc', type=float, required=False, default=0.1, help='The capacity value for controlling the Kullback-Leibler divergence (KLD) of Fc.')
    parser.add_argument('--FcLimit', type=float, required=False, default=0.05, help='The upper threshold value for frequency.')
    parser.add_argument('--DecayH', type=float, required=False, default=0.00, help='The decay effect on the cutoff frequency when creating a high-pass filter.')
    parser.add_argument('--DecayL', type=float, required=False, default=0.00, help='The decay effect on the cutoff frequency when creating a low-pass filter.')
    
    # Other parameters
    parser.add_argument('--Patience', type=int, required=False, default=300, help='The patience value for early stopping during model training.')
    
    
    #### -----------------------------------------------------   Experiment setting   -------------------------------------------------------------------------    
    ### Model related parameters
    args = parser.parse_args() # Parse the arguments
    
    LatDim = args.LatDim
    SigType = args.SigType
    CompSize = args.CompSize
    
    assert CompSize in [i for i in range(100, 1000, 100)], "Value should be one of " +str([i for i in range(100, 1000, 100)])
    
    MaskingRate = args.MaskingRate
    NoiseStd = args.NoiseStd
    MaskStd = args.MaskStd
    ReparaStd = args.ReparaStd
    Capacity_Z = args.Capacity_Z
    Capacity_Fc = args.Capacity_Fc
    FcLimit = args.FcLimit
    DecayH = args.DecayH
    DecayL = args.DecayL
    Patience = args.Patience
    

    SavePath = './Results/'
    ModelName = 'Base_'+str(SigType)+'_Z'+str(LatDim)+'_Comp'+str(CompSize)+'.hdf5'
    
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
        
    ### Model checkpoint
    ModelSaveName = SavePath+ModelName
    

    
    #### -----------------------------------------------------   Data load and processing   -------------------------------------------------------------------------    
    TrData = np.load('./Data/ProcessedData/Tr'+str(SigType)+'.npy')
    ValData = np.load('./Data/ProcessedData/Val'+str(SigType)+'.npy')
    SigDim = TrData.shape[1]
        
    
    #### -----------------------------------------------------   Model   -------------------------------------------------------------------------    
    EncModel = Encoder(SigDim=SigDim, LatDim= LatDim, Type = '', MaskingRate = MaskingRate, NoiseStd = NoiseStd, MaskStd = MaskStd, ReparaStd = ReparaStd, Reparam=True, FcLimit=FcLimit)
    FeatExtModel = FeatExtractor(SigDim=SigDim, CompSize = CompSize, DecayH=DecayH, DecayL=DecayL)
    FeatGenModel = FeatGenerator(SigDim=SigDim,FeatDim=FeatExtModel.output[1].shape[-1], LatDim= LatDim)
    ReconModel = Reconstructor(SigDim=SigDim, FeatDim=FeatExtModel.output[1].shape[-1])

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
    Beta_MI = Lossweight(name='Beta_MI', InitVal=0.05)(FeatGenOut)
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
    Z_Sampled, Z_Log_Sigma = SigBandRepModel.get_layer('Z_Mu').output, SigBandRepModel.get_layer('Z_Log_Sigma').output
    kl_Loss_Z = 0.5 * tf.reduce_sum( Z_Sampled**2  +  tf.exp(Z_Log_Sigma)- Z_Log_Sigma-1, axis=1)    
    kl_Loss_Z = tf.reduce_mean(kl_Loss_Z )
    kl_Loss_Z = Beta_Z * tf.abs(kl_Loss_Z - Capacity_Z)

    ### KL Divergence for p(FCs) vs q(FCs)
    BernP = 0.5 # hyperparameter
    FCs = SigBandRepModel.get_layer('FC_Mu').output 
    kl_Loss_FC = tf.math.log(FCs) - tf.math.log(BernP) + tf.math.log(1-FCs) - tf.math.log(1-BernP) 
    kl_Loss_FC = tf.reduce_mean(-kl_Loss_FC )
    kl_Loss_FC = Beta_Fc * tf.abs(kl_Loss_FC - Capacity_Fc)

    SigBandRepModel.add_loss(kl_Loss_Z )
    SigBandRepModel.add_metric(kl_Loss_Z , 'kl_Loss_Z')

    SigBandRepModel.add_loss(kl_Loss_FC )
    SigBandRepModel.add_metric(kl_Loss_FC , 'kl_Loss_FC')

    ## Model Compile
    SigBandRepModel.compile(optimizer='adam') 


    ### Loss and KLD_Beta controller
    #KLD_Beta_Z = KLAnneal(TargetLossName=['val_FeatRecLoss', 'val_RecLoss'], Threshold=0.001, BetaName='Beta_Z',  MaxBeta=0.1 , MinBeta=0.1, AnnealEpoch=1, UnderLimit=1e-7, verbose=2)
    #KLD_Beta_Fc = KLAnneal(TargetLossName=['val_FeatRecLoss', 'val_RecLoss'], Threshold=0.001, BetaName='Beta_Fc',  MaxBeta=0.005 , MinBeta=0.005, AnnealEpoch=1, UnderLimit=1e-7, verbose=1)
    
    RelLossDic = { 'val_ReconOut_ext':'Beta_Rec_ext', 'val_FeatRecLoss':'Beta_Feat', 'val_kl_Loss_Z':'Beta_Z', 'val_kl_Loss_FC':'Beta_Fc'}
    ScalingDic = { 'val_ReconOut_ext':200., 'val_FeatRecLoss':300., 'val_kl_Loss_Z':0.1, 'val_kl_Loss_FC':0.1}
    MinLimit = {'Beta_Rec_ext':1., 'Beta_Feat':1., 'Beta_Z':0.01, 'Beta_Fc':0.01}
    MaxLimit = {'Beta_Rec_ext':500., 'Beta_Feat':500., 'Beta_Z':0.25, 'Beta_Fc':0.25}
    RelLoss = RelLossWeight(BetaList=RelLossDic, LossScaling= ScalingDic, MinLimit= MinLimit, MaxLimit = MaxLimit, ToSaveLoss=['val_FeatRecLoss', 'val_ReconOut_ext'] , SaveWay='max' , SavePath = ModelSaveName)
    
    
       
    
    # Model Training
    #SigBandRepModel.load_weights(ModelSaveName)
    SigBandRepModel.fit(TrData, batch_size=3000, epochs=2000, shuffle=True, validation_data =(ValData, None) , callbacks=[  RelLoss]) 


