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
from Benchmarks.Models.BaseModels import *
from Utilities.Utilities import *
from Utilities.EvaluationModules import *
from Models.BenchmarkModels import *

import warnings
warnings.filterwarnings('ignore')


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
    ### Model related parameters
    SigType = ConfigSet[ConfigName]['SigType']
    LatDim = ConfigSet[ConfigName]['LatDim']
    ReparaStd = ConfigSet[ConfigName]['ReparaStd']
    
    
    ### Other parameters
    BatSize = ConfigSet[ConfigName]['BatSize']
    NEpochs = ConfigSet[ConfigName]['NEpochs']
    
    
    SavePath = './Results/'
    ModelName = ConfigName+'.hdf5'
    
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
        
    ### Model checkpoint
    ModelSaveName = SavePath+ModelName
    

    
    #### -----------------------------------------------------   Data load and processing   --------------------------------------------------------
    TrData = np.load('../Data/ProcessedData/Tr'+str(SigType)+'.npy').astype('float32')
    ValData = np.load('../Data/ProcessedData/Val'+str(SigType)+'.npy').astype('float32')
    SigDim = TrData.shape[1]
    NData = TrData.shape[0]     
        
    
    
    #### -----------------------------------------------------   Model   -------------------------------------------------------------------------    
    # ModelName selection
    if 'BaseVAE' in ConfigName:
        BenchModel = BaseVAE(SigDim, ConfigSet[ConfigName], LatDim=LatDim,  ReparaStd=ReparaStd, Reparam=True)
        TrInp, ValInp = TrData, ValData
    
    elif 'TCVAE' in ConfigName:
        BenchModel = TCVAE(SigDim, NData, ConfigSet[ConfigName], LatDim=LatDim, ReparaStd=ReparaStd, Reparam=True)
        TrInp, ValInp = TrData, ValData
    
    elif 'FACVAE' in ConfigName:
        BenchModel = FACVAE( SigDim, ConfigSet[ConfigName], LatDim=LatDim,  ReparaStd=ReparaStd, Reparam=True)
        TrInp, ValInp = TrData, ValData
    
    elif 'ConVAE' in ConfigName:
        BenchModel, Tr_Cond, Val_Cond = ConVAE(TrData, ValData,  SigDim, ConfigSet[ConfigName], LatDim=LatDim,  ReparaStd=ReparaStd, Reparam=True)
        TrInp = [TrData, Tr_Cond]
        ValInp = [ValData, Val_Cond]
        
    else:
        assert False, "Please verify if the model name is right or not."    
    
        
    
    #### ------------------------------------------------ Dynamic controller for common losses and betas ------------------------------------------------ 
    #The relative size of the loss is reflected in the weight to minimize the loss.
    
    ### Parameters for constant losse weights
    WRec = ConfigSet[ConfigName]['WRec']
    WZ = ConfigSet[ConfigName]['WZ']
    
    ### Parameters for dynamic controller for losse weights
    MnWRec = ConfigSet[ConfigName]['MnWRec']
    MnWZ = ConfigSet[ConfigName]['MnWZ']
    MxWRec = ConfigSet[ConfigName]['MxWRec']
    MxWZ = ConfigSet[ConfigName]['MxWZ']
    
    RelLossDic = { 'val_ReconOutLoss':'Beta_Rec', 'val_kl_Loss_Z':'Beta_Z' }
    ScalingDic = { 'val_ReconOutLoss':WRec, 'val_kl_Loss_Z':WZ}
    MinLimit = {'Beta_Rec':MnWRec,  'Beta_Z':MnWZ}
    MaxLimit = {'Beta_Rec':MxWRec,  'Beta_Z':MxWZ}
    RelLoss = RelLossWeight(BetaList=RelLossDic, LossScaling= ScalingDic, MinLimit= MinLimit, MaxLimit = MaxLimit, ToSaveLoss=['val_ReconOutLoss'] , SaveWay='max' , SavePath = ModelSaveName)
    
    
        
    #### ------------------------------------------------ Dynamic controller for specific losses and betas ------------------------------------------------
    if 'TCVAE' in ConfigName :
        RelLossDic['val_kl_Loss_TC'] = 'Beta_TC'
        ScalingDic['val_kl_Loss_TC'] = ConfigSet[ConfigName]['WTC']
        MinLimit['Beta_TC'] = ConfigSet[ConfigName]['MnWTC']
        MaxLimit['Beta_TC'] = ConfigSet[ConfigName]['MxWTC']
        
        RelLossDic['val_kl_Loss_MI'] = 'Beta_MI'
        ScalingDic['val_kl_Loss_MI'] = ConfigSet[ConfigName]['WMI']
        MinLimit['Beta_MI'] = ConfigSet[ConfigName]['MnWMI']
        MaxLimit['Beta_MI'] = ConfigSet[ConfigName]['MxWMI']
    
    if 'FACVAE' in ConfigName :
        RelLossDic['val_kl_Loss_TC'] = 'Beta_TC'
        ScalingDic['val_kl_Loss_TC'] = ConfigSet[ConfigName]['WTC']
        MinLimit['Beta_TC'] = ConfigSet[ConfigName]['MnWTC']
        MaxLimit['Beta_TC'] = ConfigSet[ConfigName]['MxWTC']
        
        RelLossDic['val_kl_Loss_DTC'] = 'Beta_DTC'
        ScalingDic['val_kl_Loss_DTC'] = ConfigSet[ConfigName]['WDTC']
        MinLimit['Beta_DTC'] = ConfigSet[ConfigName]['MnWDTC']
        MaxLimit['Beta_DTC'] = ConfigSet[ConfigName]['MxWDTC']
        
    
    
    
    #### Model Training
    if Resume == True:
        BenchModel.load_weights(ModelSaveName)
    BenchModel.fit(TrInp, batch_size=BatSize, epochs=NEpochs, shuffle=True, validation_data =(ValInp, None) , callbacks=[  RelLoss]) 

    