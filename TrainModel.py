import os
import sys
import numpy as np
import random
import pandas as pd
from argparse import ArgumentParser
import re

import tensorflow as tf
from tensorflow.keras import Model
from Utilities.Utilities import ReadYaml, LoadModelConfigs
from Models.Caller import *


# Refer to the execution code        
# python .\TrainModel.py --Config TCMIDKZFC_II_50_500 --GPUID 0 --Resume True    

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
    
    # Creating the parser
    parser = ArgumentParser()
    
    # Add Experiment-related parameters
    parser.add_argument('--Config', type=str, required=True, help='Set the name of the configuration to load (the name of the config in the YAML file).')
    parser.add_argument('--GPUID', type=int, required=False, default=0)
    parser.add_argument('--Resume', type=str2bool, required=False, default=False)
    
    # Parsing the arguments
    args = parser.parse_args() 
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
                gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024*24))]  
            )
        except RuntimeError as e:
            print(e)          
    

    
    #### -----------------------------------------------------   Experiment setting   -------------------------------------------------------------------------    
    
    # Loading the model configurations
    ModelConfigSet, ModelSavePath = LoadModelConfigs(ConfigName, Training=False)
    
    ### Model-related parameters
    SigType = ModelConfigSet['SigType']
    DataSource = ModelConfigSet['DataSource']
    
    ### Loss-related parameters
    LossType = ModelConfigSet['LossType']
    SpecLosses = ModelConfigSet['SpecLosses']
    
    ### Ancillary parameters
    BatSize = ModelConfigSet['BatSize']
    NEpochs = ModelConfigSet['NEpochs']
    

    
    #### -----------------------------------------------------   Data load and processing   -------------------------------------------------------------------------    
    TrData = np.load('./Data/ProcessedData/'+str(DataSource)+'Tr'+str(SigType)+'.npy')
    ValData = np.load('./Data/ProcessedData/'+str(DataSource)+'Val'+str(SigType)+'.npy')
    SigDim = TrData.shape[1]
    DataSize= TrData.shape[0]
       
    
    
    #### -----------------------------------------------------   Defining model structure -------------------------------------------------------------------------    
    # Calling Modesl
    SigRepModel = ModelCall (ModelConfigSet, SigDim, DataSize, Resume=Resume, Reparam=True, ModelSaveName=ModelSavePath)
   
    # Calling dynamic controller for losses (DCL)
    ## The relative size of the loss is reflected in the weight to minimize the loss.
    RelLoss = DCLCall (ModelConfigSet, ModelSavePath, ToSaveLoss=None, SaveWay='max', Resume=Resume, Patience=300)
    NEpochs -= (RelLoss.StartEpoch )
    
    # Model Training
    SigRepModel.fit(TrData, batch_size=BatSize, epochs=NEpochs, shuffle=True, validation_data =(ValData, None) , callbacks=[RelLoss]) 


