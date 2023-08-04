import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, GRU, Dense, Masking, Reshape, Flatten, RepeatVector, TimeDistributed, Bidirectional, Activation, GaussianNoise, Lambda, LSTM
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from Benchmarks.Models.BaseVAE import *
from Utilities.Utilities import *
from Utilities.EvaluationModules import *





def ConVAE (TrData, ValData,  SigDim, ConfigSpec, LatDim=50, SlidingSize = 50, ReparaStd=0.1, Reparam=True):
    
    ### Loss-related parameters
    Capacity_Z = ConfigSpec['Capacity_Z']
    

    #### -----------------------------------------------------   Model   -------------------------------------------------------------------------    
    ## Identifying conditions based on cumulative Power Spectral Entropy (PSE) over each frequency
    Tr_Cond = FFT_PSD(TrData, 'None')[:, 0]
    Val_Cond = FFT_PSD(ValData, 'None')[:, 0]
    
    EncModel = Encoder(SigDim=SigDim,CondDim=Tr_Cond.shape[-1], SlidingSize = SlidingSize, LatDim= LatDim, Reparam = Reparam, ReparaStd=ReparaStd)
    ReconModel = Decoder(SigDim=SigDim,CondDim=Tr_Cond.shape[-1], SlidingSize = SlidingSize, LatDim= LatDim)

    ## Model core parts
    ReconOut =ReconModel([EncModel.output, EncModel.input[-1]])

    ### Define the model
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

   
    return CVAEModel, Tr_Cond, Val_Cond