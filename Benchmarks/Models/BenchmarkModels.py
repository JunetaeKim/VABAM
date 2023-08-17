import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, GRU, Dense, Masking, Reshape, Flatten, RepeatVector, TimeDistributed, Bidirectional, Activation, GaussianNoise, Lambda, LSTM
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from Benchmarks.Models.BaseModels import *
from Models.Discriminator import FacDiscriminator
from Utilities.Utilities import Lossweight
from Utilities.AncillaryFunctions import LogNormalDensity, SplitBatch, FFT_PSD




def SKLDZ (Z_Mu, Z_Log_Sigma, Beta_Z, Capacity_Z):
    
    kl_Loss_Z = 0.5 * tf.reduce_sum( Z_Mu**2  +  tf.exp(Z_Log_Sigma)- Z_Log_Sigma-1, axis=1)    
    kl_Loss_Z = tf.reduce_mean(kl_Loss_Z )
    kl_Loss_Z = Beta_Z * tf.abs(kl_Loss_Z - Capacity_Z)
    
    return kl_Loss_Z


def BaseVAE (SigDim, ConfigSpec, LatDim=50, SlidingSize = 50, ReparaStd=0.1, Reparam=True):
    
    #### -----------------------------------------------------   Model   -------------------------------------------------------------------------    
    EncModel = Encoder(SigDim=SigDim, SlidingSize = 50, LatDim= LatDim, Reparam = True)
    ReconModel = Decoder(SigDim=SigDim, SlidingSize = 50, LatDim= LatDim)
    
    ## Model core parts
    ReconOut =ReconModel(EncModel.output)

    ### Define the model
    VAEModel = Model(EncModel.input, ReconOut)
    
    
    
    #### -----------------------------------------------------   Losses   -------------------------------------------------------------------------
    ### Loss-related parameters
    Capacity_Z = ConfigSpec['Capacity_Z']
        
    ### Weight controller; Apply beta and capacity 
    Beta_Z = Lossweight(name='Beta_Z', InitVal=0.01)(VAEModel.input)
    Beta_Rec = Lossweight(name='Beta_Rec', InitVal=1.)(VAEModel.input)

    ### Adding the RecLoss; 
    MSE = tf.keras.losses.MeanSquaredError()

    ReconOutLoss = Beta_Rec * MSE(ReconOut, EncModel.input)
    VAEModel.add_loss(ReconOutLoss)
    VAEModel.add_metric(ReconOutLoss, 'ReconOutLoss')
    print('ReconOutLoss added')


    ### KL Divergence for q(Z) vs q(Z)
    Z_Mu, Z_Log_Sigma = VAEModel.get_layer('Z_Mu').output, VAEModel.get_layer('Z_Log_Sigma').output
    kl_Loss_Z = SKLDZ(Z_Mu, Z_Log_Sigma, Beta_Z, Capacity_Z)
    VAEModel.add_loss(kl_Loss_Z )
    VAEModel.add_metric(kl_Loss_Z , 'kl_Loss_Z')
    
    
    ### Model Compile
    VAEModel.compile(optimizer='adam') 
    VAEModel.summary()

   
    return VAEModel





def ConVAE (SigDim, CondDim, ConfigSpec, LatDim=50, SlidingSize = 50, ReparaStd=0.1, Reparam=True):
    
    #### -----------------------------------------------------   Model   -------------------------------------------------------------------------    
    EncModel = Encoder(SigDim=SigDim,CondDim=CondDim, SlidingSize = SlidingSize, LatDim= LatDim, Reparam = Reparam, ReparaStd=ReparaStd)
    ReconModel = Decoder(SigDim=SigDim,CondDim=CondDim, SlidingSize = SlidingSize, LatDim= LatDim)

    ## Model core parts
    ReconOut =ReconModel([EncModel.output, EncModel.input[-1]])

    ### Define the model
    CVAEModel = Model(EncModel.input, ReconOut)
    
    
    
    #### -----------------------------------------------------   Losses   -------------------------------------------------------------------------
    ### Loss-related parameters
    Capacity_Z = ConfigSpec['Capacity_Z']
    
    ### Weight controller; Apply beta and capacity 
    Beta_Z = Lossweight(name='Beta_Z', InitVal=1.0)(ReconOut)
    Beta_Rec = Lossweight(name='Beta_Rec', InitVal=1.)(ReconOut)

    ### Adding the RecLoss; 
    MSE = tf.keras.losses.MeanSquaredError()
    ReconOutLoss = Beta_Rec * MSE(ReconOut, EncModel.input[0])

    ### Adding losses to the model
    CVAEModel.add_loss(ReconOutLoss)
    CVAEModel.add_metric(ReconOutLoss, 'ReconOutLoss')
    
    ### KL Divergence for q(Z) vs q(Z)
    Z_Mu, Z_Log_Sigma = CVAEModel.get_layer('Z_Mu').output, CVAEModel.get_layer('Z_Log_Sigma').output
    kl_Loss_Z = SKLDZ(Z_Mu, Z_Log_Sigma, Beta_Z, Capacity_Z)
    CVAEModel.add_loss(kl_Loss_Z )
    CVAEModel.add_metric(kl_Loss_Z , 'kl_Loss_Z')

    
    ### Model Compile
    CVAEModel.compile(optimizer='adam') 
    CVAEModel.summary()

   
    return CVAEModel




def TCVAE (  SigDim, NData, ConfigSpec, LatDim=50, SlidingSize = 50, ReparaStd=0.1, Reparam=True):
    
    #### -----------------------------------------------------   Model   -------------------------------------------------------------------------    
    EncModel = Encoder(SigDim=SigDim, SlidingSize = 50, LatDim= LatDim, Reparam = Reparam)
    ReconModel = Decoder(SigDim=SigDim, SlidingSize = 50, LatDim= LatDim)

    ## Model core parts
    ReconOut =ReconModel(EncModel.output)

    ### Define the total model
    TCVAEModel = Model(EncModel.input, ReconOut)

    
   
    #### -----------------------------------------------------   Losses   -------------------------------------------------------------------------
    ### Loss-related parameters
    Capacity_Z = ConfigSpec['Capacity_Z']
    Capacity_TC = ConfigSpec['Capacity_TC']
    Capacity_MI = ConfigSpec['Capacity_MI']
    BatSize = ConfigSpec['BatSize']
    
    ### Weight controller; Apply beta and capacity 
    Beta_Z = Lossweight(name='Beta_Z', InitVal=1.0)(ReconOut)
    Beta_TC = Lossweight(name='Beta_TC', InitVal=1.0)(ReconOut)
    Beta_MI = Lossweight(name='Beta_MI', InitVal=1.0)(ReconOut)
    Beta_Rec = Lossweight(name='Beta_Rec', InitVal=1.)(ReconOut)
    
    
    ### Adding the RecLoss; 
    MSE = tf.keras.losses.MeanSquaredError()
    ReconOutLoss = Beta_Rec * MSE(ReconOut, EncModel.input)
    
    
    'Reference: https://github.com/YannDubs/disentangling-vae/issues/60#issuecomment-705164833' 
    'https://github.com/JunetaeKim/disentangling-vae-torch/blob/master/disvae/utils/math.py#L54'
    ### KL Divergence for q(Z) vs q(Z)_Prod
    Z_Mu, Z_Log_Sigma, Zs = TCVAEModel.get_layer('Z_Mu').output, TCVAEModel.get_layer('Z_Log_Sigma').output, TCVAEModel.get_layer('Zs').output
    LogProb_QZ = LogNormalDensity(Zs[:, None], Z_Mu[None], Z_Log_Sigma[None])
    Log_QZ = tf.reduce_logsumexp(tf.reduce_sum(LogProb_QZ, axis=2),   axis=1) - tf.math.log(BatSize * NData * 1.)
    Log_QZ_Prod = tf.reduce_sum( tf.reduce_logsumexp(LogProb_QZ, axis=1) - tf.math.log(BatSize * NData * 1.),   axis=1)
    kl_Loss_TC = tf.reduce_mean(Log_QZ - Log_QZ_Prod)
    kl_Loss_TC = Beta_TC * tf.abs(kl_Loss_TC - Capacity_TC)

    
    ### MI Loss ; I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
    Log_QZX = tf.reduce_sum(LogNormalDensity(Zs, Z_Mu, Z_Log_Sigma), axis=1)
    kl_Loss_MI = tf.reduce_mean((Log_QZX - Log_QZ))
    kl_Loss_MI = Beta_MI * tf.abs(kl_Loss_MI - Capacity_MI)

    
    ### KL Divergence for p(Z) vs q(Z) # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
    Log_PZ = tf.reduce_sum(LogNormalDensity(Zs, 0., 0.), axis=1)
    DW_kl_Loss_Z = tf.reduce_mean(Log_QZ_Prod - Log_PZ)
    kl_Loss_Z = Beta_Z * tf.abs(DW_kl_Loss_Z - Capacity_Z)
    
    
    ### Adding losses to the model
    TCVAEModel.add_loss(ReconOutLoss)
    TCVAEModel.add_metric(ReconOutLoss, 'ReconOutLoss')
    
    TCVAEModel.add_loss(kl_Loss_Z )
    TCVAEModel.add_metric(kl_Loss_Z , 'kl_Loss_Z')

    TCVAEModel.add_loss(kl_Loss_TC )
    TCVAEModel.add_metric(kl_Loss_TC , 'kl_Loss_TC')
    
    TCVAEModel.add_loss(kl_Loss_MI )
    TCVAEModel.add_metric(kl_Loss_MI , 'kl_Loss_MI')

    
    ### Model Compile
    TCVAEModel.compile(optimizer='adam') 
    TCVAEModel.summary()

   
    return TCVAEModel




def FACVAE (  SigDim, ConfigSpec, LatDim=50, SlidingSize = 50, ReparaStd=0.1, Reparam=True):
    
    ### Model related parameters
    DiscHiddenSize = ConfigSpec['DiscHiddenSize']
    
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

    ### Define the model
    FactorVAEModel = Model(EncModel.input, [FacDiscOut_D1, ReconOut])
    
    
    
    #### -----------------------------------------------------   Losses   -------------------------------------------------------------------------
    ### Loss-related parameters
    Capacity_Z = ConfigSpec['Capacity_Z']
    Capacity_TC = ConfigSpec['Capacity_TC']
    Capacity_DTC = ConfigSpec['Capacity_DTC']
    
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
    kl_Loss_Z = SKLDZ(Z_Mu_D1, Z_Log_Sigma_D1, Beta_Z, Capacity_Z)

    
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

   
    return FactorVAEModel