import os
import sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import yaml

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


def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    

if __name__ == "__main__":

    
    # Create the parser
    parser = ArgumentParser()
    
    # Add Model related parameters
    parser.add_argument('--Config', type=str, required=True, help='Set the name of the configuration to load (the name of the config in the YAML file).')

    yaml_path = './Config/Config.yml'
    ConfigSet = read_yaml(yaml_path)

    #### -----------------------------------------------------   Experiment setting   -------------------------------------------------------------------------    
    ### Model related parameters
    args = parser.parse_args() # Parse the arguments
    
    ConfigName = args.Config
    
    SigType = ConfigSet[ConfigName]['SigType']
    LatDim = ConfigSet[ConfigName]['LatDim']
    CompSize = ConfigSet[ConfigName]['CompSize']
    
    MaskingRate = ConfigSet[ConfigName]['MaskingRate']
    NoiseStd = ConfigSet[ConfigName]['NoiseStd']
    MaskStd = ConfigSet[ConfigName]['MaskStd']
    ReparaStd = ConfigSet[ConfigName]['ReparaStd']
    Capacity_Z = ConfigSet[ConfigName]['Capacity_Z']
    Capacity_Fc = ConfigSet[ConfigName]['Capacity_Fc']
    FcLimit = ConfigSet[ConfigName]['FcLimit']
    DecayH = ConfigSet[ConfigName]['DecayH']
    DecayL = ConfigSet[ConfigName]['DecayL']
    
    Patience = ConfigSet[ConfigName]['Patience']
    WRec = ConfigSet[ConfigName]['WRec']
    WFeat = ConfigSet[ConfigName]['WFeat']
    WZ = ConfigSet[ConfigName]['WZ']
    WFC = ConfigSet[ConfigName]['WFC']
    WTC = ConfigSet[ConfigName]['WTC']
    WMI = ConfigSet[ConfigName]['WMI']
    
    ### Other parameters
    MnWRec = ConfigSet[ConfigName]['MnWRec']
    MnWFeat = ConfigSet[ConfigName]['MnWFeat']
    MnWZ = ConfigSet[ConfigName]['MnWZ']
    MnWFC = ConfigSet[ConfigName]['MnWFC']
    MnWTC = ConfigSet[ConfigName]['MnWTC']
    MnWMI = ConfigSet[ConfigName]['MnWMI']
    
    MxWRec = ConfigSet[ConfigName]['MxWRec']
    MxWFeat = ConfigSet[ConfigName]['MxWFeat']
    MxWZ = ConfigSet[ConfigName]['MxWZ']
    MxWFC = ConfigSet[ConfigName]['MxWFC']
    MxWTC = ConfigSet[ConfigName]['MxWTC']
    MxWMI = ConfigSet[ConfigName]['MxWMI']
    

    SavePath = './Results/'
    ModelName = ConfigName+'_'+SigType+'.hdf5'
    
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
    Beta_Z = Lossweight(name='Beta_Z', InitVal=1.0)(FeatGenOut)
    Beta_Fc = Lossweight(name='Beta_Fc', InitVal=1.0)(FeatGenOut)
    Beta_TC = Lossweight(name='Beta_TC', InitVal=1.0)(FeatGenOut)
    Beta_MI = Lossweight(name='Beta_MI', InitVal=1.0)(FeatGenOut)
    Beta_Rec_ext = Lossweight(name='Beta_Rec_ext', InitVal=1.)(FeatGenOut)
    Beta_Rec_gen = Lossweight(name='Beta_Rec_gen', InitVal=1.)(FeatGenOut)
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

    ### KL Divergence for p(FCs) vs q(FCs)
    BernP = 0.5 # hyperparameter
    FCs = SigBandRepModel.get_layer('FC_Mu').output 
    kl_Loss_FC = tf.math.log(FCs) - tf.math.log(BernP) + tf.math.log(1-FCs) - tf.math.log(1-BernP) 
    kl_Loss_FC = tf.reduce_mean(-kl_Loss_FC )
    kl_Loss_FC = Beta_Fc * tf.abs(kl_Loss_FC - Capacity_Fc)

    
    ### KL Divergence for q(Z) vs q(Z)_Prod
    Z_Mu, Z_Log_Sigma, Zs = SigBandRepModel.get_layer('Z_Mu').output, SigBandRepModel.get_layer('Z_Log_Sigma').output, SigBandRepModel.get_layer('Zs').output
    LogProb_QZ = LogNormalDensity(Zs[:, None], Z_Mu[None], Z_Log_Sigma[None])
    Log_QZ_Prod = tf.reduce_sum( tf.reduce_logsumexp(LogProb_QZ, axis=1, keepdims=False),   axis=1,  keepdims=False)
    Log_QZ = tf.reduce_logsumexp(tf.reduce_sum(LogProb_QZ, axis=2, keepdims=False),   axis=1,   keepdims=False)
    kl_Loss_TC = -tf.reduce_mean(Log_QZ - Log_QZ_Prod)
    kl_Loss_TC = Beta_TC * kl_Loss_TC

    ### MI Loss ; I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
    Log_QZX = tf.reduce_sum(LogNormalDensity(Zs, Z_Mu, Z_Log_Sigma), axis=1)
    kl_Loss_MI = -tf.reduce_mean((Log_QZX - Log_QZ))
    kl_Loss_MI = Beta_MI * kl_Loss_MI

    ### KL Divergence for p(Z) vs q(Z) # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
    Log_PZ = tf.reduce_sum(LogNormalDensity(Zs, 0., 0.), axis=1)
    DW_kl_Loss_Z = tf.reduce_mean(Log_QZ_Prod - Log_PZ)
    kl_Loss_Z = Beta_Z * tf.abs(DW_kl_Loss_Z - Capacity_Z)
    
    
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


