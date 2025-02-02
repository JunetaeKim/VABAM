import sys
# setting path
sys.path.append('../')
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from argparse import ArgumentParser
from Utilities.Utilities import *
from Benchmarks.Models.BenchmarkCaller import *


import warnings
warnings.filterwarnings('ignore')


# Refer to the execution code
# python .\TrainBenchmark.py --Config FACVAE_ART_30 --GPUID 0

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def reset_all_seeds(seed=7):
    """Function to reset random seeds"""
    random.seed(seed)         # Set Python's built-in random seed
    np.random.seed(seed)      # Set NumPy seed
    tf.random.set_seed(seed)  # Set TensorFlow seed
    os.environ['PYTHONHASHSEED'] = str(seed)  # Fix Python hash seed
    #os.environ['TF_DETERMINISTIC_OPS'] = '1'  




if __name__ == "__main__":

    reset_all_seeds(seed=0)    
    
    # Create the parser
    parser = ArgumentParser()
    
    # Add Experiment-related parameters
    parser.add_argument('--Config', type=str, required=True, help='Set the name of the configuration to load (the name of the config in the YAML file).')
    parser.add_argument('--GPUID', type=int, required=False, default=0)
    parser.add_argument('--Resume', type=str2bool, required=False, default=False)
    
    args = parser.parse_args() # Parse the arguments
    ConfigName = args.Config
    GPU_ID = args.GPUID
    Resume = args.Resume
    
    
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
    elif 'II' in ConfigName:
        LoadConfig = 'Config' + 'II'
        SubPath = 'II/'
    else:
        assert False, "Please verify if the data type is properly included in the name of the configuration. The configuration name should be structured as 'Config' + 'data type', such as ConfigART."

    yaml_path = './Config/'+LoadConfig+'.yml'
    ConfigSet = ReadYaml(yaml_path)

    
    #### -----------------------------------------------------   Experiment setting   -------------------------------------------------------------------------    
    ### Model related parameters
    SigType = ConfigSet['Common_Param']['SigType']
    DataSource = ConfigSet['Models'][ConfigName]['DataSource']
    CommonParams = ConfigSet['Common_Param']
    ModelParams = ConfigSet["Models"][ConfigName]
    
    ### Other parameters
    BatSize = ConfigSet['Models'][ConfigName]['BatSize']
    NEpochs = ConfigSet['Models'][ConfigName]['NEpochs']
    
    
    ### Experiment setting
    SavePath = './Results/'
    ModelName = ConfigName+'.hdf5'
    
    if not os.path.exists(SavePath+SubPath):
        os.makedirs(SavePath+SubPath)
        
    ### Model checkpoint
    ModelSaveName = SavePath+SubPath+ModelName
    

    
    #### -----------------------------------------------------   Data load and processing   --------------------------------------------------------
    if 'Wavenet' in ConfigName:
        SlidingSize = ConfigSet['Models'][ConfigName]['SlidingSize']
    
        TrRaw = np.load('../Data/ProcessedData/'+str(DataSource)+'Tr'+str(SigType)+'.npy')
        ValRaw = np.load('../Data/ProcessedData/'+str(DataSource)+'Val'+str(SigType)+'.npy')
    
        TrSampled = np.load('../Data/ProcessedData/Sampled'+str(DataSource)+'Tr'+str(SigType)+'.npy').astype('float32') # Sampled_TrData
        ValSampled = np.load('../Data/ProcessedData/Sampled'+str(DataSource)+'Val'+str(SigType)+'.npy').astype('float32') # Sampled_ValData
        TrOut = np.load('../Data/ProcessedData/MuLaw'+str(DataSource)+'Tr'+str(SigType)+'.npy').astype('int32') # MuLaw_TrData
        ValOut = np.load('../Data/ProcessedData/MuLaw'+str(DataSource)+'Val'+str(SigType)+'.npy').astype('int32') # MuLaw_ValData

        TrInp = [TrSampled, TrRaw]
        ValInp = [ValSampled, ValRaw]
        
        ModelParams['DataSize'] = TrSampled.shape[0] 
        ModelParams['SigDim'] = TrSampled.shape[1]
        
    else:
        TrInp = np.load('../Data/ProcessedData/'+str(DataSource)+'Tr'+str(SigType)+'.npy')
        ValInp = np.load('../Data/ProcessedData/'+str(DataSource)+'Val'+str(SigType)+'.npy')
        ModelParams['DataSize'] = TrInp.shape[0] 
        ModelParams['SigDim'] = TrInp.shape[1]

    
    
    #### -----------------------------------------------------  Defining model structure -------------------------------------------------------------------------     
    # Calling Modesl
    BenchModel, TrInp, ValInp = ModelCall ({**CommonParams, **ModelParams}, ConfigName, TrInp, ValInp, Resume=Resume, Reparam=True, ModelSaveName=ModelSaveName) 


    #### -----------------------------------------------------  Execute model training -------------------------------------------------------------------------     
    if 'VAE' in ConfigName:
        # Calling dynamic controller for losses (DCL)
        ## The relative size of the loss is reflected in the weight to minimize the loss.
        RelLoss = DCLCall ({**CommonParams, **ModelParams}, ConfigName, ModelSaveName, ToSaveLoss=None, SaveWay='max', Resume=Resume)
        NEpochs -= (RelLoss.StartEpoch )    
        
        #### Model Training
        BenchModel.fit(TrInp, batch_size=BatSize, epochs=NEpochs, shuffle=True, validation_data =(ValInp, None) , callbacks=[RelLoss]) 
        
    else: 
        # Checkpoint callback
        CheckPoint = tf.keras.callbacks.ModelCheckpoint(filepath=ModelSaveName, save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min', verbose=1)
        # Early-stop callback
        EarlyStop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True )
        # Train model
        BenchModel.fit(TrInp, TrOut, validation_data=(ValInp, ValOut), epochs=ModelParams['NEpochs'], batch_size=ModelParams['BatSize'],  callbacks=[CheckPoint, EarlyStop])

    