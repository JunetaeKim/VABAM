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
from Utilities.EvaluationModules import *


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
    
    ### Loss-related parameters
    Capacity_Z = ConfigSet[ConfigName]['Capacity_Z']
    
    ### Other parameters
    BatSize = ConfigSet[ConfigName]['BatSize']
    NEpochs = ConfigSet[ConfigName]['NEpochs']
    
    ### Parameters for constant losse weights
    WRec = ConfigSet[ConfigName]['WRec']
    WZ = ConfigSet[ConfigName]['WZ']
    
    ### Parameters for dynamic controller for losse weights
    MnWRec = ConfigSet[ConfigName]['MnWRec']
    MnWZ = ConfigSet[ConfigName]['MnWZ']
    MxWRec = ConfigSet[ConfigName]['MxWRec']
    MxWZ = ConfigSet[ConfigName]['MxWZ']

    
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
        
        
    
    #### -----------------------------------------------------   Model   -------------------------------------------------------------------------    
    ## Identifying conditions based on cumulative Power Spectral Entropy (PSE) over each frequency
    Tr_Cond = FFT_PSE(TrData, 'None')[:, 0]
    Val_Cond = FFT_PSE(ValData, 'None')[:, 0]
    
    EncModel = Encoder(SigDim=SigDim,CondDim=Tr_Cond.shape[-1], SlidingSize = 50, LatDim= LatDim, Reparam = True)
    ReconModel = Decoder(SigDim=SigDim,CondDim=Tr_Cond.shape[-1], SlidingSize = 50, LatDim= LatDim)

    ## Model core parts
    ReconOut =ReconModel([EncModel.output, EncModel.input[-1]])

    ### Define the total model
    CVAEModel = Model(EncModel.input, ReconOut)
    
    
    
    #### -----------------------------------------------------   Losses   -------------------------------------------------------------------------
    ### Weight controller; Apply beta and capacity 
    Beta_Z = Lossweight(name='Beta_Z', InitVal=1.0)(ReconOut)
    Beta_Rec = Lossweight(name='Beta_Rec', InitVal=1.)(ReconOut)

    ### Adding the RecLoss; 
    MSE = tf.keras.losses.MeanSquaredError()
    ReconOutLoss = Beta_Rec * MSE(ReconOut, EncModel.input[0])

    ### KL Divergence for q(Z) vs q(Z)
    Z_Mu, Z_Log_Sigma, Zs = CVAEModel.get_layer('Z_Mu').output, CVAEModel.get_layer('Z_Log_Sigma').output, CVAEModel.get_layer('Zs').output
    kl_Loss_Z = 0.5 * tf.reduce_sum( Z_Mu**2  +  tf.exp(Z_Log_Sigma)- Z_Log_Sigma-1, axis=1)    
    kl_Loss_Z = tf.reduce_mean(kl_Loss_Z )
    kl_Loss_Z = Beta_Z * tf.abs(kl_Loss_Z - Capacity_Z)
    
    
    ### Adding losses to the model
    CVAEModel.add_loss(ReconOutLoss)
    CVAEModel.add_metric(ReconOutLoss, 'ReconOutLoss')
    
    CVAEModel.add_loss(kl_Loss_Z )
    CVAEModel.add_metric(kl_Loss_Z , 'kl_Loss_Z')
    
    
    ### Model Compile
    CVAEModel.compile(optimizer='adam') 
    CVAEModel.summary()


    ### Dynamic controller for common losses and betas; The relative size of the loss is reflected in the weight to minimize the loss.
    RelLossDic = { 'val_ReconOutLoss':'Beta_Rec', 'val_kl_Loss_Z':'Beta_Z' }
    ScalingDic = { 'val_ReconOutLoss':WRec, 'val_kl_Loss_Z':WZ}
    MinLimit = {'Beta_Rec':MnWRec,  'Beta_Z':MnWZ}
    MaxLimit = {'Beta_Rec':MxWRec,  'Beta_Z':MxWZ}
    RelLoss = RelLossWeight(BetaList=RelLossDic, LossScaling= ScalingDic, MinLimit= MinLimit, MaxLimit = MaxLimit, ToSaveLoss=['val_ReconOutLoss'] , SaveWay='max' , SavePath = ModelSaveName)
    
    
    ### Model Training
    if Resume == True:
        CVAEModel.load_weights(ModelSaveName)
    CVAEModel.fit([TrData, Tr_Cond], batch_size=BatSize, epochs=NEpochs, shuffle=True, validation_data =([ValData, Val_Cond], None) , callbacks=[  RelLoss]) 

    