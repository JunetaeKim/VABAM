import sys
# setting path
sys.path.append('../')
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from Utilities.Utilities import *
from Benchmarks.Models.BenchmarkCaller import *
import shutil

import warnings
warnings.filterwarnings('ignore')


# Refer to the execution code
# python .\FineTuneBenchmark.py --Config TCVAE_ART_50 --GPUID 0 --Restore True
   

if __name__ == "__main__":

    
    # Create the parser
    parser = ArgumentParser()
    
    # Add Experiment-related parameters
    parser.add_argument('--Config', type=str, required=True, help='Set the name of the configuration to load (the name of the config in the YAML file).')
    parser.add_argument('--GPUID', type=int, required=False, default=0)
    parser.add_argument('--Restore', type=bool, required=False, default=False)
    
    # Parsing the arguments
    args = parser.parse_args() 
    ConfigName = args.Config
    GPU_ID = args.GPUID
    Restore = args.Restore
    Resume = True
    
    
    ## GPU selection
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_ID)

    # TensorFlow memory configuration
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            gpu = gpus[0]  # Fix the index as zero since GPU_ID has already been given. 
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration
            (
                gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024*23.5))]  
            )
        except RuntimeError as e:
            print(e)        
    
    
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
    

    #### --------------------------------------- Restore the original files (i.e., initial training results) ---------------------------------------
    if Restore == True:
        if 'ART' in SigType:
            Sig ='/ART/'
        elif 'II' in SigType: 
            Sig ='/II/'
        
        OriginResName = [Name for Name in os.listdir('./Results' +Sig+ 'Original') if  ConfigName in Name]
        OriginLogName = [Name for Name in os.listdir('./Logs' +Sig+ 'Original') if  ConfigName in Name]
        
        # Copy the file, overwriting if it exists
        shutil.copy('./Results' + Sig + 'Original/' + OriginResName[0], './Results' + Sig  + OriginResName[0])
        print(OriginResName, ' is restored')
        shutil.copy('./Logs' + Sig + 'Original/' + OriginLogName[0], './Logs' + Sig  + OriginLogName[0])
        print(OriginLogName, ' is restored')

    
    
    #### -----------------------------------------------------   Data load and processing   --------------------------------------------------------
    TrData = np.load('../Data/ProcessedData/Tr'+str(SigType)+'.npy').astype('float32')
    ValData = np.load('../Data/ProcessedData/Val'+str(SigType)+'.npy').astype('float32')
    
    
    
    #### -----------------------------------------------------  Defining model structure -------------------------------------------------------------------------     
    # Calling Modesl
    BenchModel, TrInp, ValInp = ModelCall (ConfigSet[ConfigName], ConfigName, TrData, ValData, Resume=Resume, Reparam=True, ModelSaveName=ModelSaveName) 
    BenchModel.compile(optimizer='adam') 
    
    # Calling dynamic controller for losses (DCL)
    ## The relative size of the loss is reflected in the weight to minimize the loss.
    RelLoss = DCLCall (ConfigSet[ConfigName], ConfigName, ModelSaveName, ToSaveLoss=None, SaveWay='max', Resume=Resume, Buffer=5)
    NEpochs -= (RelLoss.StartEpoch )    
    
    #### Model Training
    BenchModel.fit(TrInp, batch_size=BatSize, epochs=NEpochs, shuffle=True, validation_data =(ValInp, None) , callbacks=[  RelLoss]) 

    