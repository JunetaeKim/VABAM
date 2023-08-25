import sys
# setting path
sys.path.append('../')
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from Utilities.Utilities import *
from Benchmarks.Models.BenchmarkCaller import *


import warnings
warnings.filterwarnings('ignore')


   

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
        SubPath = 'ART/'
    elif 'PLETH' in ConfigName:
        LoadConfig = 'Config' + 'PLETH'
        SubPath = 'PLETH/'
    elif 'II' in ConfigName:
        LoadConfig = 'Config' + 'II'
        SubPath = 'II/'
    else:
        assert False, "Please verify if the data type is properly included in the name of the configuration. The configuration name should be structured as 'Config' + 'data type', such as ConfigART."

    yaml_path = './Config/'+LoadConfig+'.yml'
    ConfigSet = ReadYaml(yaml_path)

    
    #### -----------------------------------------------------   Experiment setting   -------------------------------------------------------------------------    
    ### Model related parameters
    SigType = ConfigSet[ConfigName]['SigType']
    LatDim = ConfigSet[ConfigName]['LatDim']
    
    
    ### Other parameters
    BatSize = ConfigSet[ConfigName]['BatSize']
    NEpochs = ConfigSet[ConfigName]['NEpochs']
    
    
    ### Experiment setting
    SavePath = './Results/'
    ModelName = ConfigName+'.hdf5'
    
    if not os.path.exists(SavePath+SubPath):
        os.makedirs(SavePath+SubPath)
        
    ### Model checkpoint
    ModelSaveName = SavePath+SubPath+ModelName
    

    
    #### -----------------------------------------------------   Data load and processing   --------------------------------------------------------
    TrData = np.load('../Data/ProcessedData/Tr'+str(SigType)+'.npy').astype('float32')
    ValData = np.load('../Data/ProcessedData/Val'+str(SigType)+'.npy').astype('float32')
    
    
    
    #### -----------------------------------------------------  Defining model structure -------------------------------------------------------------------------     
    # Calling Modesl
    BenchModel, TrInp, ValInp = ModelCall (ConfigSet[ConfigName], ConfigName, TrData, ValData, Resume=Resume, Reparam=True, ModelSaveName=ModelSaveName) 
    
    # Calling dynamic controller for losses (DCL)
    ## The relative size of the loss is reflected in the weight to minimize the loss.
    RelLoss = DCLCall (ConfigSet[ConfigName], ConfigName, ModelSaveName, ToSaveLoss=None, SaveWay='max')
        
    
    #### Model Training
    BenchModel.fit(TrInp, batch_size=BatSize, epochs=NEpochs, shuffle=True, validation_data =(ValInp, None) , callbacks=[  RelLoss]) 

    