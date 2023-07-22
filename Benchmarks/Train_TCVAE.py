import sys
# setting path
sys.path.append('../')
import os

import numpy as np
import pandas as pd
from argparse import ArgumentParser
import yaml

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, GRU, Dense, Masking, Reshape, Flatten, RepeatVector, TimeDistributed, Bidirectional, Activation, GaussianNoise, Lambda, LSTM
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from Benchmarks.Libs.BaseVAE.BaseVAE import *
from Utilities.Utilities import *


def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    

    
if __name__ == "__main__":

    
    # Create the parser
    parser = ArgumentParser()
    
    # Add Experiment-related parameters
    parser.add_argument('--Config', type=str, required=True, help='Set the name of the configuration to load (the name of the config in the YAML file).')
    parser.add_argument('--GPUID', type=int, required=False, default=1)
    parser.add_argument('--Resume', type=bool, required=False, default=False)
    
    args = parser.parse_args() # Parse the arguments
    ConfigName = args.Config
    GPU_ID = args.GPUID
    Resume = args.Resume
    
    
    ## GPU selection
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_ID)

    # TensorFlow wizardry
    config = tf.compat.v1.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    # Create a session with the above options specified.
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))         
    
    
    if 'ART' in ConfigName:
        LoadConfig = 'Config' + 'ART'
    elif 'PLETH' in ConfigName:
        LoadConfig = 'Config' + 'PLETH'
    elif 'II' in ConfigName:
        LoadConfig = 'Config' + 'II'
    else:
        assert False, "Please verify if the data type is properly included in the name of the configuration. The configuration name should be structured as 'Config' + 'data type', such as ConfigART."

    yaml_path = './Config/'+LoadConfig+'.yml'
    ConfigSet = read_yaml(yaml_path)

    #### -----------------------------------------------------   Experiment setting   -------------------------------------------------------------------------    
    ### Model-related parameters
    SigType = ConfigSet[ConfigName]['SigType']
    LatDim = ConfigSet[ConfigName]['LatDim']
    
    MaskingRate = ConfigSet[ConfigName]['MaskingRate']
    NoiseStd = ConfigSet[ConfigName]['NoiseStd']
    MaskStd = ConfigSet[ConfigName]['MaskStd']
    ReparaStd = ConfigSet[ConfigName]['ReparaStd']
    
    ### Loss-related parameters
    Capacity_Z = ConfigSet[ConfigName]['Capacity_Z']
    Capacity_TC = ConfigSet[ConfigName]['Capacity_TC']
    Capacity_MI = ConfigSet[ConfigName]['Capacity_MI']

    ### Other parameters
    BatSize = ConfigSet[ConfigName]['BatSize']
    NEpochs = ConfigSet[ConfigName]['NEpochs']
        
    ### Parameters for constant losse weights
    WRec = ConfigSet[ConfigName]['WRec']
    WZ = ConfigSet[ConfigName]['WZ']
    WTC = ConfigSet[ConfigName]['WTC']
    WMI = ConfigSet[ConfigName]['WMI']
    
    ### Parameters for dynamic controller for losse weights
    MnWRec = ConfigSet[ConfigName]['MnWRec']
    MnWZ = ConfigSet[ConfigName]['MnWZ']
    MnWTC = ConfigSet[ConfigName]['MnWTC']
    MnWMI = ConfigSet[ConfigName]['MnWMI']
    
    MxWRec = ConfigSet[ConfigName]['MxWRec']
    MxWZ = ConfigSet[ConfigName]['MxWZ']
    MxWTC = ConfigSet[ConfigName]['MxWTC']
    MxWMI = ConfigSet[ConfigName]['MxWMI']
    

    SavePath = './Results/'
    ModelName = ConfigName+'_'+SigType+'.hdf5'
    
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
        
    ### Model checkpoint
    ModelSaveName = SavePath+ModelName
    

    
    #### -----------------------------------------------------   Data load and processing   -------------------------------------------------------------------------    
    TrData = np.load('../Data/ProcessedData/Tr'+str(SigType)+'.npy').astype('float32')
    ValData = np.load('../Data/ProcessedData/Val'+str(SigType)+'.npy').astype('float32')
    SigDim = TrData.shape[1]
    NData = TrData.shape[0] 
        
    
    #### -----------------------------------------------------   Model   -------------------------------------------------------------------------    
    EncModel = Encoder(SigDim=SigDim, SlidingSize = 50, LatDim= LatDim, Reparam = True)
    ReconModel = Decoder(SigDim=SigDim, SlidingSize = 50, LatDim= LatDim)

    ## Model core parts
    ReconOut =ReconModel(EncModel.output)

    ### Define the total model
    TCVAEModel = Model(EncModel.input, ReconOut)

    ### Weight controller; Apply beta and capacity 
    Beta_Z = Lossweight(name='Beta_Z', InitVal=1.0)(ReconOut)
    Beta_TC = Lossweight(name='Beta_TC', InitVal=1.0)(ReconOut)
    Beta_MI = Lossweight(name='Beta_MI', InitVal=1.0)(ReconOut)
    Beta_Rec = Lossweight(name='Beta_Rec', InitVal=1.)(ReconOut)

    ### Adding the RecLoss; 
    MSE = tf.keras.losses.MeanSquaredError()
    
    ReconOutLoss = Beta_Rec * MSE(ReconOut, EncModel.input)
    TCVAEModel.add_loss(ReconOutLoss)
    TCVAEModel.add_metric(ReconOutLoss, 'ReconOutLoss')
    
    
    'Reference: https://github.com/YannDubs/disentangling-vae/issues/60#issuecomment-705164833' 
    'https://github.com/JunetaeKim/disentangling-vae-torch/blob/master/disvae/utils/math.py#L54'
    ### KL Divergence for q(Z) vs q(Z)_Prod
    Z_Mu, Z_Log_Sigma, Zs = TCVAEModel.get_layer('Z_Mu').output, TCVAEModel.get_layer('Z_Log_Sigma').output, TCVAEModel.get_layer('Zs').output
    LogProb_QZ = LogNormalDensity(Zs[:, None], Z_Mu[None], Z_Log_Sigma[None])
    Log_QZ = tf.reduce_logsumexp(tf.reduce_sum(LogProb_QZ, axis=2),   axis=1) - tf.math.log(BatSize * NData * 1.)
    Log_QZ_Prod = tf.reduce_sum( tf.reduce_logsumexp(LogProb_QZ, axis=1) - tf.math.log(BatSize * NData * 1.),   axis=1)

    '''
    # use stratification
    log_iw_mat = log_importance_weight_matrix(BatSize, NData)
    log_iw_mat = tf.cast(log_iw_mat, Zs.dtype)  
    Log_QZ = tf.reduce_logsumexp(log_iw_mat + tf.reduce_sum(LogProb_QZ, axis=2), axis=1)         
    Log_QZ_Prod =  tf.reduce_sum(tf.reduce_logsumexp(tf.reshape(log_iw_mat, [BatSize, BatSize, 1])+LogProb_QZ, axis=1), axis=1)
    '''
    kl_Loss_TC = tf.reduce_mean(Log_QZ - Log_QZ_Prod)
    kl_Loss_TC = Beta_TC * tf.abs(kl_Loss_TC - Capacity_TC)

    
    ### MI Loss ; I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
    Log_QZX = tf.reduce_sum(LogNormalDensity(Zs, Z_Mu, Z_Log_Sigma), axis=1)
    kl_Loss_MI = tf.reduce_mean((Log_QZX - Log_QZ))
    kl_Loss_MI = Beta_MI * tf.abs(kl_Loss_MI - Capacity_MI)

    
    ### KL Divergence for p(Z) vs q(Z) # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
    Log_PZ = tf.reduce_sum(LogNormalDensity(Zs, 0., 0.), axis=1)
    DW_kl_Loss_Z = tf.reduce_mean(Log_QZ_Prod - Log_PZ)
    kl_Loss_Z = Beta_Z * tf.abs(DW_kl_Loss_Z - Capacity_Z)
    
    
    
    TCVAEModel.add_loss(kl_Loss_Z )
    TCVAEModel.add_metric(kl_Loss_Z , 'kl_Loss_Z')

    TCVAEModel.add_loss(kl_Loss_TC )
    TCVAEModel.add_metric(kl_Loss_TC , 'kl_Loss_TC')
    
    TCVAEModel.add_loss(kl_Loss_MI )
    TCVAEModel.add_metric(kl_Loss_MI , 'kl_Loss_MI')

    
    ## Model Compile
    TCVAEModel.compile(optimizer='adam') 
    TCVAEModel.summary()



    RelLossDic = { 'val_ReconOutLoss':'Beta_Rec', 'val_kl_Loss_Z':'Beta_Z', 'val_kl_Loss_TC':'Beta_TC', 'val_kl_Loss_MI':'Beta_MI'}
    ScalingDic = { 'val_ReconOutLoss':WRec, 'val_kl_Loss_Z':WZ,  'val_kl_Loss_TC':WTC, 'val_kl_Loss_MI':WMI}
    MinLimit = {'Beta_Rec':MnWRec,  'Beta_Z':MnWZ,  'Beta_TC':MnWTC, 'Beta_MI':MnWMI}
    MaxLimit = {'Beta_Rec':MxWRec,  'Beta_Z':MxWZ, 'Beta_TC':MxWTC, 'Beta_MI':MxWMI}
    RelLoss = RelLossWeight(BetaList=RelLossDic, LossScaling= ScalingDic, MinLimit= MinLimit, MaxLimit = MaxLimit, ToSaveLoss=['val_ReconOutLoss'] , SaveWay='max' , SavePath = ModelSaveName)
    
    
    
    # Model Training
    if Resume == True:
        TCVAEModel.load_weights(ModelSaveName)
    TCVAEModel.fit(TrData, batch_size=BatSize, epochs=NEpochs, shuffle=True, validation_data =(ValData, None) , callbacks=[  RelLoss]) 


