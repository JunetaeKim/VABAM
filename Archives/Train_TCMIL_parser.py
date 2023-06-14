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
    parser.add_argument('--WRec', type=float, required=False, default=200, help='the weight for the reconstruction loss')
    parser.add_argument('--WFeat', type=float, required=False, default=500, help='the weight for the feature loss')
    parser.add_argument('--WZ', type=float, required=False, default=0.1, help='the weight for the Z loss')
    parser.add_argument('--WFC', type=float, required=False, default=0.1, help='the weight for the FC loss')
    parser.add_argument('--WTC', type=float, required=False, default=0.1, help='the weight for the TC loss')
    parser.add_argument('--WMI', type=float, required=False, default=0.1, help='the weight for the MI loss')
    
    parser.add_argument('--MnWRec', type=float, required=False, default=1, help='the min weight for the reconstruction loss')
    parser.add_argument('--MnWFeat', type=float, required=False, default=1, help='the min weight for the feature loss')
    parser.add_argument('--MnWZ', type=float, required=False, default=0.05, help='the min weight for the Z loss')
    parser.add_argument('--MnWFC', type=float, required=False, default=0.05, help='the min weight for the FC loss')
    parser.add_argument('--MnWTC', type=float, required=False, default=0.01, help='the min weight for the TC loss')
    parser.add_argument('--MnWMI', type=float, required=False, default=0.01, help='the min weight for the MI loss')   

    parser.add_argument('--MxWRec', type=float, required=False, default=1000, help='the max weight for the reconstruction loss')
    parser.add_argument('--MxWFeat', type=float, required=False, default=1000, help='the max weight for the feature loss')
    parser.add_argument('--MxWZ', type=float, required=False, default=0.5, help='the max weight for the Z loss')
    parser.add_argument('--MxWFC', type=float, required=False, default=0.4, help='the max weight for the FC loss')
    parser.add_argument('--MxWTC', type=float, required=False, default=0.25, help='the max weight for the TC loss')
    parser.add_argument('--MxWMI', type=float, required=False, default=0.25, help='the max weight for the MI loss')   


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
    WRec = args.WRec
    WFeat = args.WFeat
    WZ = args.WZ
    WFC = args.WFC
    WTC = args.WTC
    WMI = args.WMI
    
    ### Other parameters
    MnWRec = args.MnWRec
    MnWFeat = args.MnWFeat
    MnWZ = args.MnWZ
    MnWFC = args.MnWFC
    MnWTC = args.MnWTC
    MnWMI = args.MnWMI
    
    MxWRec = args.MxWRec
    MxWFeat = args.MxWFeat
    MxWZ = args.MxWZ
    MxWFC = args.MxWFC
    MxWTC = args.MxWTC
    MxWMI = args.MxWMI
    

    SavePath = './Results/'
    ModelName = 'TCMIL_'+str(SigType)+'_Z'+str(LatDim)+'_Comp'+str(CompSize)+'.hdf5'
    
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

    ### MI Loss ; I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
    Log_QZX = tf.reduce_sum(LogNormalDensity(Zs, Z_Mu, Z_Log_Sigma), axis=1)
    kl_Loss_MI = -tf.reduce_mean((Log_QZX - Log_QZ))
    kl_Loss_MI = Beta_MI * kl_Loss_MI

    
    
    SigBandRepModel.add_loss(kl_Loss_Z )
    SigBandRepModel.add_metric(kl_Loss_Z , 'kl_Loss_Z')

    SigBandRepModel.add_loss(kl_Loss_FC )
    SigBandRepModel.add_metric(kl_Loss_FC , 'kl_Loss_FC')

    SigBandRepModel.add_loss(kl_Loss_TC )
    SigBandRepModel.add_metric(kl_Loss_TC , 'kl_Loss_TC')
    
    SigBandRepModel.add_loss(kl_Loss_MI )
    SigBandRepModel.add_metric(kl_Loss_MI , 'kl_Loss_MI')

    
    ## Model Compile
    SigBandRepModel.compile(optimizer='adam') 
    SigBandRepModel.summary()


    ### Loss and KLD_Beta controller
    #KLD_Beta_Z = KLAnneal(TargetLossName=['val_FeatRecLoss', 'val_RecLoss'], Threshold=0.001, BetaName='Beta_Z',  MaxBeta=0.1 , MinBeta=0.1, AnnealEpoch=1, UnderLimit=1e-7, verbose=2)
    #KLD_Beta_Fc = KLAnneal(TargetLossName=['val_FeatRecLoss', 'val_RecLoss'], Threshold=0.001, BetaName='Beta_Fc',  MaxBeta=0.005 , MinBeta=0.005, AnnealEpoch=1, UnderLimit=1e-7, verbose=1)

    RelLossDic = { 'val_ReconOut_ext':'Beta_Rec_ext', 'val_FeatRecLoss':'Beta_Feat', 'val_kl_Loss_Z':'Beta_Z', 'val_kl_Loss_FC':'Beta_Fc', 'val_kl_Loss_TC':'Beta_TC', 'val_kl_Loss_MI':'Beta_MI'}
    ScalingDic = { 'val_ReconOut_ext':WRec, 'val_FeatRecLoss':WFeat, 'val_kl_Loss_Z':WZ, 'val_kl_Loss_FC':WFC, 'val_kl_Loss_TC':WTC, 'val_kl_Loss_MI':WMI}
    MinLimit = {'Beta_Rec_ext':MnWRec, 'Beta_Feat':MnWFeat, 'Beta_Z':MnWZ, 'Beta_Fc':MnWFC, 'Beta_TC':MnWTC, 'Beta_MI':MnWMI}
    MaxLimit = {'Beta_Rec_ext':MxWRec, 'Beta_Feat':MxWFeat, 'Beta_Z':MxWZ, 'Beta_Fc':MxWFC, 'Beta_TC':MxWTC, 'Beta_MI':MxWMI}
    RelLoss = RelLossWeight(BetaList=RelLossDic, LossScaling= ScalingDic, MinLimit= MinLimit, MaxLimit = MaxLimit, ToSaveLoss=['val_FeatRecLoss', 'val_ReconOut_ext'] , SaveWay='max' , SavePath = ModelSaveName)
    
    
    
    # Model Training
    #SigBandRepModel.load_weights(ModelSaveName)
    SigBandRepModel.fit(TrData, batch_size=3000, epochs=4000, shuffle=True, validation_data =(ValData, None) , callbacks=[  RelLoss]) 


