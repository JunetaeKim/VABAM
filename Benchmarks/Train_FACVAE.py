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
from Models.Discriminator import FacDiscriminator
from Utilities.Utilities import *

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
    DiscHiddenSize = ConfigSet[ConfigName]['DiscHiddenSize']
    ReparaStd = ConfigSet[ConfigName]['ReparaStd']
    
    ### Loss-related parameters
    Capacity_Z = ConfigSet[ConfigName]['Capacity_Z']
    Capacity_TC = ConfigSet[ConfigName]['Capacity_TC']
    Capacity_DTC = ConfigSet[ConfigName]['Capacity_DTC']
    
    ### Other parameters
    BatSize = ConfigSet[ConfigName]['BatSize']
    NEpochs = ConfigSet[ConfigName]['NEpochs']
        
    ### Parameters for constant losse weights
    WRec = ConfigSet[ConfigName]['WRec']
    WZ = ConfigSet[ConfigName]['WZ']
    WTC = ConfigSet[ConfigName]['WTC']
    WDTC = ConfigSet[ConfigName]['WDTC']
    
    ### Parameters for dynamic controller for losse weights
    MnWRec = ConfigSet[ConfigName]['MnWRec']
    MnWZ = ConfigSet[ConfigName]['MnWZ']
    MnWTC = ConfigSet[ConfigName]['MnWTC']
    MnWDTC = ConfigSet[ConfigName]['MnWDTC']
    
    MxWRec = ConfigSet[ConfigName]['MxWRec']
    MxWZ = ConfigSet[ConfigName]['MxWZ']
    MxWTC = ConfigSet[ConfigName]['MxWTC']
    MxWDTC = ConfigSet[ConfigName]['MxWDTC']
    

    SavePath = './Results/'
    ModelName = ConfigName+'_'+SigType+'.hdf5'
    
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
        
    ### Model checkpoint
    ModelSaveName = SavePath+ModelName
    

    
    #### -----------------------------------------------------   Data load and processing   -----------------------------------------------------------------    
    TrData = np.load('../Data/ProcessedData/Tr'+str(SigType)+'.npy').astype('float32')
    ValData = np.load('../Data/ProcessedData/Val'+str(SigType)+'.npy').astype('float32')
    SigDim = TrData.shape[1]
        
    
    #### -----------------------------------------------------   Model   -------------------------------------------------------------------------    
    EncModel = Encoder(SigDim=SigDim, SlidingSize = 50, LatDim= LatDim, Reparam = True)
    ReconModel = Decoder(SigDim=SigDim, SlidingSize = 50, LatDim= LatDim)
    FacDiscModel = FacDiscriminator(LatDim, DiscHiddenSize)

    ## Batch split 
    BatchSize = tf.shape(EncModel.input)[0]
    HalfBatchIdx1 = tf.range(BatchSize//2)
    HalfBatchIdx2 = tf.range(BatchSize//2, BatchSize)
    Z_D1, Z_D2 = SplitBatch(EncModel.output, HalfBatchIdx1, HalfBatchIdx2, mode='Both')

    ## Model core parts
    ReconOut = ReconModel(EncModel.output)
    FacDiscOut_D1 =  FacDiscModel(Z_D1)

    ### Define the total model
    FactorVAEModel = Model(EncModel.input, [FacDiscOut_D1, ReconOut])

    
    
    #### -----------------------------------------------------   Losses   -------------------------------------------------------------------------
    ### Weight controller; Apply beta and capacity 
    Beta_Z = Lossweight(name='Beta_Z', InitVal=0.01)(FactorVAEModel.input)
    Beta_TC = Lossweight(name='Beta_TC', InitVal=0.01)(FactorVAEModel.input)
    Beta_DTC = Lossweight(name='Beta_DTC', InitVal=0.01)(FactorVAEModel.input)
    Beta_Rec = Lossweight(name='Beta_Rec', InitVal=1.)(FactorVAEModel.input)

    
    ### Adding the RecLoss; 
    MSE = tf.keras.losses.MeanSquaredError()
    ReconOutLoss = Beta_Rec * MSE(ReconOut, EncModel.input)

    
    ### KL Divergence for p(Z) vs q(Z)
    Z_Mu, Z_Log_Sigma, Zs = FactorVAEModel.get_layer('Z_Mu').output, FactorVAEModel.get_layer('Z_Log_Sigma').output, FactorVAEModel.get_layer('Zs').output
    Z_Mu_D1 = SplitBatch(Z_Mu, HalfBatchIdx1, HalfBatchIdx2, mode='D1')
    Z_Log_Sigma_D1 = SplitBatch(Z_Log_Sigma, HalfBatchIdx1, HalfBatchIdx2, mode='D1')

    kl_Loss_Z = 0.5 * tf.reduce_sum( Z_Mu_D1**2  +  tf.exp(Z_Log_Sigma_D1)- Z_Log_Sigma_D1-1, axis=1)    
    kl_Loss_Z = tf.reduce_mean(kl_Loss_Z )
    kl_Loss_Z = Beta_Z * tf.abs(kl_Loss_Z - Capacity_Z)


    ### Total Correlation # KL(q(z)||prod_j q(z_j)); log(p_true/p_false) = logit_true - logit_false
    kl_Loss_TC = tf.reduce_mean(FacDiscOut_D1[:, 0] - FacDiscOut_D1[:, 1])
    kl_Loss_TC = Beta_TC * tf.abs(kl_Loss_TC - Capacity_TC)


    ### Discriminator Loss
    Batch2_Size = tf.shape(Z_D2)[0]
    PermZ_D2 = tf.concat([tf.nn.embedding_lookup(Z_D2[:, i][:,None], tf.random.shuffle(tf.range(Batch2_Size))) for i in range(Z_D1.shape[-1])], axis=1)
    PermZ_D2 = tf.stop_gradient(PermZ_D2) # 
    FacDiscOut_D2 = FacDiscModel(PermZ_D2)

    Ones = tf.ones_like(HalfBatchIdx1)[:,None]
    Zeros = tf.zeros_like(HalfBatchIdx2)[:,None]
    CCE = tf.keras.losses.MeanSquaredError()

    kl_Loss_DTC = 0.5 * (CCE(Zeros, FacDiscOut_D1) + CCE(Ones, FacDiscOut_D2))
    kl_Loss_DTC = Beta_DTC * tf.abs(kl_Loss_DTC - Capacity_DTC)

    
    ### Adding losses to the model
    FactorVAEModel.add_loss(ReconOutLoss)
    FactorVAEModel.add_metric(ReconOutLoss, 'ReconOutLoss')
    
    FactorVAEModel.add_loss(kl_Loss_Z )
    FactorVAEModel.add_metric(kl_Loss_Z , 'kl_Loss_Z')
    
    FactorVAEModel.add_loss(kl_Loss_TC )
    FactorVAEModel.add_metric(kl_Loss_TC , 'kl_Loss_TC')
    
    FactorVAEModel.add_loss(kl_Loss_DTC )
    FactorVAEModel.add_metric(kl_Loss_DTC , 'kl_Loss_DTC')

    
    ### Model Compile
    FactorVAEModel.compile(optimizer='adam') 
    FactorVAEModel.summary()


    ### Dynamic controller for common losses and betas; The relative size of the loss is reflected in the weight to minimize the loss.
    RelLossDic = { 'val_ReconOutLoss':'Beta_Rec', 'val_kl_Loss_Z':'Beta_Z', 'val_kl_Loss_TC':'Beta_TC', 'val_kl_Loss_DTC':'Beta_DTC'}
    ScalingDic = { 'val_ReconOutLoss':WRec, 'val_kl_Loss_Z':WZ,  'val_kl_Loss_TC':WTC, 'val_kl_Loss_DTC':WDTC}
    MinLimit = {'Beta_Rec':MnWRec,  'Beta_Z':MnWZ,  'Beta_TC':MnWTC, 'Beta_DTC':MnWDTC}
    MaxLimit = {'Beta_Rec':MxWRec,  'Beta_Z':MxWZ, 'Beta_TC':MxWTC, 'Beta_DTC':MxWDTC}
    RelLoss = RelLossWeight(BetaList=RelLossDic, LossScaling= ScalingDic, MinLimit= MinLimit, MaxLimit = MaxLimit, ToSaveLoss=['val_ReconOutLoss'] , SaveWay='max' , SavePath = ModelSaveName)
    
    
    
    # Model Training
    if Resume == True:
        FactorVAEModel.load_weights(ModelSaveName)
    FactorVAEModel.fit(TrData, batch_size=BatSize, epochs=NEpochs, shuffle=True, validation_data =(ValData, None) , callbacks=[  RelLoss]) 


