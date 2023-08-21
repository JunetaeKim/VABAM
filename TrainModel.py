import os
import sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import re

import tensorflow as tf
from tensorflow.keras import Model
from Utilities.Utilities import ReadYaml
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

    YamlPath = './Config/'+LoadConfig+'.yml'
    ConfigSet = ReadYaml(YamlPath)

    
    #### -----------------------------------------------------   Experiment setting   -------------------------------------------------------------------------    
    ### Model-related parameters
    SigType = ConfigSet[ConfigName]['SigType']
    
    ### Loss-related parameters
    LossType = ConfigSet[ConfigName]['LossType']
    SpecLosses = ConfigSet[ConfigName]['SpecLosses']
    
    ### Ancillary parameters
    BatSize = ConfigSet[ConfigName]['BatSize']
    NEpochs = ConfigSet[ConfigName]['NEpochs']
    
    
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
   
    # Calling dynamic controller for losses (DCL)
    ## The relative size of the loss is reflected in the weight to minimize the loss.
    RelLoss = DCLCall (ConfigSet[ConfigName], ModelSaveName, ToSaveLoss=None, SaveWay='max')
    
    
    # Model Training
    SigRepModel.fit(TrData, batch_size=BatSize, epochs=NEpochs, shuffle=True, validation_data =(ValData, None) , callbacks=[RelLoss]) 


