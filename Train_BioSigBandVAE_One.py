import os
import sys
import numpy as np
import pandas as pd


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, GRU, Dense, Masking, Reshape, Flatten, RepeatVector, TimeDistributed, Bidirectional, Activation, GaussianNoise, Lambda, LSTM
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from Models.BioSigBandVAE_One import *
from Utilities.Utilities import *


## GPU selection
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# TensorFlow wizardry
config = tf.compat.v1.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.98
# Create a session with the above options specified.
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))     


if __name__ == "__main__":

    #### -----------------------------------------------------   Experiment setting   -------------------------------------------------------------------------    
    SavePath = './Results/'
    ModelName = 'SigBandRep_AllOnce_ExtRec.hdf5'
    
    ### Model related parameters
    LatDim = 3
    MaskingRate = 0.02
    NoiseStd = 0.002
    MaskStd = 0.1
    ReparaStd = 0.1
    Capacity_Z = 0.1
    Capacity_Fc = 0.6
    
    ### Other parameters
    Patience = 300
    TrRate = 0.8
    
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
        
    ### Model checkpoint
    ModelSaveName = SavePath+ModelName
    
    ### Model Early stop
    EarlyStop = EarlyStopping(monitor='val_loss', patience=Patience)
    
       
        
    
    #### -----------------------------------------------------   Data load and processing   -------------------------------------------------------------------------    
    DATA = np.load('./Data/AsanTRSet.npy')
    SigDim = DATA.shape[1]
    
    np.random.seed(7)
    PermutedDATA = np.random.permutation(DATA)
    TrLen = int(PermutedDATA.shape[0] * TrRate)

    TrData = PermutedDATA[:TrLen]
    ValData = PermutedDATA[TrLen:]
    
    
    
    
    
    #### -----------------------------------------------------   Model   -------------------------------------------------------------------------    
    ### Encoder - FeatExtractor
    EncInp, EncOut = Encoder(SigDim=SigDim, LatDim= LatDim, Type = '', MaskingRate = MaskingRate, NoiseStd = NoiseStd, MaskStd = MaskStd, ReparaStd = ReparaStd, Reparam=True)
    FeatExtOut = FeatExtractor(EncOut)

    ### Encoder - FeatGenerator - Reconstruction
    FeatGenOut = FeatGenerator([EncOut[1],EncOut[2][:, :2], EncOut[2][:, 2:]])
    FeatGenOut = ReName(FeatGenOut, 'FeatGenOut')

    ReconOut = Reconstructor([FeatExtOut , EncOut[2]], SigDim = SigDim)
    ReconOut = ReName(ReconOut, 'ReconOut')

    ### Define the total model
    SigBandRepModel = Model(EncInp, ReconOut)

    ### Weight controller; Apply beta and capacity 
    Capacity_Z = 0.1 # 0.1 0.05
    Capacity_Fc = 0.6
    Beta_Z = Lossweight(name='Beta_Z')(FeatGenOut)
    Beta_Fc = Lossweight(name='Beta_Fc')(FeatGenOut)
    Beta_Rec = Lossweight(name='Beta_Rec', InitVal=500.)(FeatGenOut)
    Beta_Feat = Lossweight(name='Beta_Feat', InitVal=500.)(FeatGenOut)


    ### Adding the RecLoss; 
    MSE = tf.keras.losses.MeanSquaredError()
    RecLoss = Beta_Rec * MSE(ReconOut, EncInp)
    SigBandRepModel.add_loss(RecLoss)
    SigBandRepModel.add_metric(RecLoss, 'RecLoss')


    ### Adding the FeatRecLoss; It allows connection between the extractor and generator
    FeatRecLoss= Beta_Feat * MSE(tf.concat(FeatGenOut, axis=-1), tf.concat(FeatExtOut, axis=-1))
    SigBandRepModel.add_loss(FeatRecLoss)
    SigBandRepModel.add_metric(FeatRecLoss, 'FeatRecLoss')

    ### KL Divergence for p(Z) vs q(Z)
    Z_Sampled, Z_Log_Sigma = SigBandRepModel.get_layer('Z_Mean').output, SigBandRepModel.get_layer('Z_Log_Sigma').output
    kl_Loss_Z = 0.5 * tf.reduce_sum( Z_Sampled**2  +  tf.exp(Z_Log_Sigma)- Z_Log_Sigma-1, axis=1)    
    kl_Loss_Z = tf.reduce_mean(kl_Loss_Z )
    kl_Loss_Z = Beta_Z * tf.abs(kl_Loss_Z - Capacity_Z)

    ### KL Divergence for p(FCs) vs q(FCs)
    BernP = 0.5 # hyperparameter
    FCs = SigBandRepModel.get_layer('FCs').output
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

    RelLossDic = {'val_RecLoss':'Beta_Rec', 'val_FeatRecLoss':'Beta_Feat', 'val_kl_Loss_Z':'Beta_Z', 'val_kl_Loss_FC':'Beta_Fc'}
    ScalingDic = {'val_RecLoss':100., 'val_FeatRecLoss':100., 'val_kl_Loss_Z':0.1, 'val_kl_Loss_FC':0.1}
    MinLimit = {'Beta_Rec':1., 'Beta_Feat':1., 'Beta_Z':0.01, 'Beta_Fc':0.01}
    MaxLimit = {'Beta_Rec':500., 'Beta_Feat':500., 'Beta_Z':0.07, 'Beta_Fc':0.05}
    RelLoss = RelLossWeight(BetaList=RelLossDic, LossScaling= ScalingDic, MinLimit= MinLimit, MaxLimit = MaxLimit, ToSaveLoss=['val_FeatRecLoss', 'val_RecLoss'] , SaveWay='max' , SavePath = ModelSaveName)
    
    
    
    
    
    # Model Training
    #SigBandRepModel.load_weights(ModelSaveName)
    SigBandRepModel.fit(TrData, batch_size=3500, epochs=650, shuffle=True, validation_data =(ValData, None) , callbacks=[EarlyStop,  RelLoss]) 

