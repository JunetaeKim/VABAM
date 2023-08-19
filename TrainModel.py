import os
import sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import re

import tensorflow as tf
from tensorflow.keras import Model
from Utilities.Utilities import *
from Models.Caller import *



    

if __name__ == "__main__":

    
    # Create the parser
    parser = ArgumentParser()
    
    # Add Experiment-related parameters
    parser.add_argument('--Config', type=str, required=True, help='Set the name of the configuration to load (the name of the config in the YAML file).')
    parser.add_argument('--GPUID', type=int, required=False, default=1)
    parser.add_argument('--Resume', type=bool, required=False, default=False)
    
    args = parser.parse_args() # Parse the arguments
    ConfigName = args.Config
    CompSize = re.findall(r'\d+', ConfigName)[-1]
    assert CompSize in [ num for i in os.listdir('./Config') for num in re.findall(r'\d+', i)], "Please chek 'CompSize' included in the ConfigName."
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
        LoadConfig = 'Config' + 'ART' + CompSize
        SubPath = 'ART/'
    elif 'PLETH' in ConfigName:
        LoadConfig = 'Config' + 'PLETH' + CompSize
        SubPath = 'PLETH/'
    elif 'II' in ConfigName:
        LoadConfig = 'Config' + 'II' + CompSize
        SubPath = 'II/'
    else:
        assert False, "Please verify if the data type is properly included in the name of the configuration. The configuration name should be structured as 'Config' + 'data type', such as ConfigART."

    yaml_path = './Config/'+LoadConfig+'.yml'
    ConfigSet = read_yaml(yaml_path)

    
    #### -----------------------------------------------------   Experiment setting   -------------------------------------------------------------------------    
    ### Model-related parameters
    SigType = ConfigSet[ConfigName]['SigType']
    
    ### Loss-related parameters
    LossType = ConfigSet[ConfigName]['LossType']
    
    ### Ancillary parameters
    BatSize = ConfigSet[ConfigName]['BatSize']
    NEpochs = ConfigSet[ConfigName]['NEpochs']
    
    ### Parameters for constant losse weights
    WRec = ConfigSet[ConfigName]['WRec']
    WFeat = ConfigSet[ConfigName]['WFeat']
    WZ = ConfigSet[ConfigName]['WZ']
    
    ### Parameters for dynamic controller for losse weights
    SpecLosses = ConfigSet[ConfigName]['SpecLosses']
    MnWRec = ConfigSet[ConfigName]['MnWRec']
    MnWFeat = ConfigSet[ConfigName]['MnWFeat']
    MnWZ = ConfigSet[ConfigName]['MnWZ']
    
    MxWRec = ConfigSet[ConfigName]['MxWRec']
    MxWFeat = ConfigSet[ConfigName]['MxWFeat']
    MxWZ = ConfigSet[ConfigName]['MxWZ']
    
    
    
    ### Experiment setting
    SavePath = './Results/'
    ModelName = ConfigName+'.hdf5'
    
    if not os.path.exists(SavePath+SubPath):
        os.mkdir(SavePath+SubPath)
        
    ### Model checkpoint
    ModelSaveName = SavePath+SubPath+ModelName
    

    
    #### -----------------------------------------------------   Data load and processing   -------------------------------------------------------------------------    
    TrData = np.load('./Data/ProcessedData/Tr'+str(SigType)+'.npy')
    ValData = np.load('./Data/ProcessedData/Val'+str(SigType)+'.npy')
    SigDim = TrData.shape[1]
    DataSize= TrData.shape[0]

    
    #### -----------------------------------------------------   Defining model structure -------------------------------------------------------------------------    
    # Calling Modesl
    SigRepModel = ModelCall (ConfigSet[ConfigName], SigDim, DataSize, Resume=Resume, Reparam=True, ModelSaveName=ModelSaveName)
   

    ### Dynamic controller for common losses and betas; The relative size of the loss is reflected in the weight to minimize the loss.
    RelLossDic = { 'val_OrigRecLoss':'Beta_Orig', 'val_FeatRecLoss':'Beta_Feat', 'val_kl_Loss_Z':'Beta_Z'}
    ScalingDic = { 'val_OrigRecLoss':WRec, 'val_FeatRecLoss':WFeat, 'val_kl_Loss_Z':WZ}
    MinLimit = {'Beta_Orig':MnWRec, 'Beta_Feat':MnWFeat, 'Beta_Z':MnWZ}
    MaxLimit = {'Beta_Orig':MxWRec, 'Beta_Feat':MxWFeat, 'Beta_Z':MxWZ}
    
    
    ### Dynamic controller for specific losses and betas
    if 'FC' in SpecLosses :
        RelLossDic['val_kl_Loss_FC'] = 'Beta_Fc'
        ScalingDic['val_kl_Loss_FC'] = ConfigSet[ConfigName]['WFC']
        MinLimit['Beta_Fc'] = ConfigSet[ConfigName]['MnWFC']
        MaxLimit['Beta_Fc'] = ConfigSet[ConfigName]['MxWFC']
        
    if 'TC' in SpecLosses :
        RelLossDic['val_kl_Loss_TC'] = 'Beta_TC'
        ScalingDic['val_kl_Loss_TC'] = ConfigSet[ConfigName]['WTC']
        MinLimit['Beta_TC'] = ConfigSet[ConfigName]['MnWTC']
        MaxLimit['Beta_TC'] = ConfigSet[ConfigName]['MxWTC']
        
    if 'MI' in SpecLosses :
        RelLossDic['val_kl_Loss_MI'] = 'Beta_MI'
        ScalingDic['val_kl_Loss_MI'] = ConfigSet[ConfigName]['WMI']
        MinLimit['Beta_MI'] = ConfigSet[ConfigName]['MnWMI']
        MaxLimit['Beta_MI'] = ConfigSet[ConfigName]['MxWMI']
        
    if LossType =='FACLosses':
        RelLossDic['val_kl_Loss_TC'] = 'Beta_TC'
        ScalingDic['val_kl_Loss_TC'] = ConfigSet[ConfigName]['WTC']
        MinLimit['Beta_TC'] = ConfigSet[ConfigName]['MnWTC']
        MaxLimit['Beta_TC'] = ConfigSet[ConfigName]['MxWTC']
        
        RelLossDic['val_kl_Loss_DTC'] = 'Beta_DTC'
        ScalingDic['val_kl_Loss_DTC'] = ConfigSet[ConfigName]['WDTC']
        MinLimit['Beta_DTC'] = ConfigSet[ConfigName]['MnWDTC']
        MaxLimit['Beta_DTC'] = ConfigSet[ConfigName]['MxWDTC']
        

    RelLoss = RelLossWeight(BetaList=RelLossDic, LossScaling= ScalingDic, MinLimit= MinLimit, MaxLimit = MaxLimit, ToSaveLoss=['val_FeatRecLoss', 'val_OrigRecLoss'] , SaveWay='max' , SavePath = ModelSaveName)
    
    
    # Model Training
    SigRepModel.fit(TrData, batch_size=BatSize, epochs=NEpochs, shuffle=True, validation_data =(ValData, None) , callbacks=[RelLoss]) 


