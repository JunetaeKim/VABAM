import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from Utilities.Utilities import Lossweight


def LogNormalDensity(LatSamp, LatMean, LogSquaScale):
    Norm = tf.math.log(2. * tf.constant(np.pi))
    InvSigma = tf.math.exp(-LogSquaScale)
    MeanSampDiff = (LatSamp - LatMean)
    return -0.5 * (MeanSampDiff * MeanSampDiff * InvSigma + LogSquaScale + Norm)



def SplitBatch (Vec, HalfBatchIdx1, HalfBatchIdx2, mode='Both'):
    
    HalfBatch1 = tf.nn.embedding_lookup(Vec, HalfBatchIdx1)
    HalfBatch2 = tf.nn.embedding_lookup(Vec, HalfBatchIdx2)
    
    if mode=='Both':
        return  HalfBatch1, HalfBatch2
    elif mode=='D1':
        return  HalfBatch1
    elif mode=='D2':
        return  HalfBatch2
    
    
    
    
    
def TCLosses (Models, DataSize, LossConfigSet, ModelType='TCMIDKZ'):
    
    
    ###-------------------------------- Model structuring -------------------------------------------- ###
    ## Model core parts
    EncModel,FeatExtModel,FeatGenModel,ReconModel =  Models
    EncInp =EncModel.input
    InpZ = EncModel.output[2]
    InpFCCommon = EncModel.output[1][:, :2]
    InpFCEach = EncModel.output[1][:, 2:]

    ## Each output of each model
    FeatExtOut = FeatExtModel(EncModel.output[:2])
    FeatGenOut = FeatGenModel([InpFCCommon, InpFCEach, InpZ])
    ReconExtOut = ReconModel(FeatExtOut)
    ReconGenOut = ReconModel(FeatGenOut)
    
    ### Define the total model
    SigRepModel = Model(EncInp, [FeatGenOut, ReconExtOut, ReconGenOut])
    
    
    
    ###-------------------------------- Weights for losses -------------------------------------------- ###
    ### Weight controller; Apply beta and capacity 
    Beta_Z = Lossweight(name='Beta_Z', InitVal=0.01)(SigRepModel.input)
    Beta_Fc = Lossweight(name='Beta_Fc', InitVal=0.01)(SigRepModel.input)
    Beta_TC = Lossweight(name='Beta_TC', InitVal=0.01)(SigRepModel.input)
    Beta_MI = Lossweight(name='Beta_MI', InitVal=0.01)(SigRepModel.input)
    Beta_Orig = Lossweight(name='Beta_Orig', InitVal=1.)(SigRepModel.input)
    Beta_Feat = Lossweight(name='Beta_Feat', InitVal=1.)(SigRepModel.input)



    ###-------------------------------- Common losses -------------------------------------------- ###

    ### Adding the RecLoss; 
    MSE = tf.keras.losses.MeanSquaredError()
    ReconOut_ext = Beta_Orig * MSE(ReconExtOut, EncInp)

    SigRepModel.add_loss(ReconOut_ext)
    SigRepModel.add_metric(ReconOut_ext, 'OrigRecLoss')
    print('OrigRecLoss added')

    ### Adding the FeatRecLoss; It allows connection between the extractor and generator
    FeatRecLoss= Beta_Feat * MSE(tf.concat(FeatGenOut, axis=-1), tf.concat(FeatExtOut, axis=-1))

    SigRepModel.add_loss(FeatRecLoss)
    SigRepModel.add_metric(FeatRecLoss, 'FeatRecLoss')
    print('FeatRecLoss added')




    ###-------------------------------- Specific losses -------------------------------------------- ###

    ### Total Correlation # KL(q(z)||prod_j q(z_j)) = log q(z) - sum_j log q(z_j)
    'Reference: https://github.com/YannDubs/disentangling-vae/issues/60#issuecomment-705164833' 
    'https://github.com/JunetaeKim/disentangling-vae-torch/blob/master/disvae/utils/math.py#L54'
    'https://github.com/rtqichen/beta-tcvae/blob/master/elbo_decomposition.py'
    Z_Mu, Z_Log_Sigma, Zs = SigRepModel.get_layer('Z_Mu').output, SigRepModel.get_layer('Z_Log_Sigma').output, SigRepModel.get_layer('Zs').output
    LogProb_QZi = tf.maximum(LogNormalDensity(Zs[:, None], Z_Mu[None], Z_Log_Sigma[None]), np.log(1/DataSize)) 
    LogProb_QZ = tf.reduce_sum(LogProb_QZi, axis=2, keepdims=False)
    JointEntropy  = tf.reduce_logsumexp(-tf.math.log(DataSize*1.) + LogProb_QZ ,   axis=1,   keepdims=False)
    MarginalEntropies = tf.reduce_sum( - tf.math.log(DataSize*1.) + tf.reduce_logsumexp(LogProb_QZi, axis=1),  axis=1)
    kl_Loss_TC = tf.reduce_mean( JointEntropy - MarginalEntropies)
    
    
            
    ### MI Loss ; I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
    Log_QZX = tf.reduce_sum(LogNormalDensity(Zs, Z_Mu, Z_Log_Sigma), axis=1)
    kl_Loss_MI = tf.reduce_mean(Log_QZX - JointEntropy)
    
    
    
    ### KL Divergence for p(FC) vs q(FC)
    BernP = 0.5 # hyperparameter
    FC_Mu = SigRepModel.get_layer('FC_Mu').output 
    kl_Loss_FC = FC_Mu*(tf.math.log(FC_Mu) - tf.math.log(BernP)) + (1-FC_Mu)*(tf.math.log(1-FC_Mu) - tf.math.log(1-BernP))
    kl_Loss_FC = tf.reduce_mean(kl_Loss_FC )
    
    
    
    ### KL Divergence for p(Z) vs q(Z)
    if 'SKZ' in ModelType: #Standard KLD for Z
        kl_Loss_Z = 0.5 * tf.reduce_sum( Z_Mu**2  +  tf.exp(Z_Log_Sigma)- Z_Log_Sigma-1, axis=1)    
        kl_Loss_Z = tf.reduce_mean(kl_Loss_Z )
        print('SKZ loss selected')
    elif 'DKZ' in ModelType: #Dimensional-wise KLD for Z  
        # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        Log_PZ = tf.reduce_sum(LogNormalDensity(Zs, 0., 0.), axis=1)
        kl_Loss_Z = tf.reduce_mean( MarginalEntropies - Log_PZ)
        print('DKZ loss selected')


        
    ### Adding specific losses
    Capacity_Z = LossConfigSet['Capacity_Z']
    kl_Loss_Z = Beta_Z * tf.abs(kl_Loss_Z - Capacity_Z)
    SigRepModel.add_loss(kl_Loss_Z )
    SigRepModel.add_metric(kl_Loss_Z , 'kl_Loss_Z')
    print('kl_Loss_Z added')
    
    if 'FC' in ModelType :
        Capacity_Fc = LossConfigSet['Capacity_Fc']
        kl_Loss_FC = Beta_Fc * tf.abs(kl_Loss_FC - Capacity_Fc)
        SigRepModel.add_loss(kl_Loss_FC )
        SigRepModel.add_metric(kl_Loss_FC , 'kl_Loss_FC')
        print('kl_Loss_FC added')
    
    if 'TC' in ModelType :
        Capacity_TC = LossConfigSet['Capacity_TC']
        kl_Loss_TC = Beta_TC * tf.abs(kl_Loss_TC - Capacity_TC)
        SigRepModel.add_loss(kl_Loss_TC )
        SigRepModel.add_metric(kl_Loss_TC , 'kl_Loss_TC')
        print('kl_Loss_TC added')

    if 'MI' in ModelType :
        Capacity_MI = LossConfigSet['Capacity_MI']
        kl_Loss_MI = Beta_MI * tf.abs(kl_Loss_MI - Capacity_MI)
        SigRepModel.add_loss(kl_Loss_MI )
        SigRepModel.add_metric(kl_Loss_MI , 'kl_Loss_MI')
        print('kl_Loss_MI added')

        
    return SigRepModel





def FACLosses (Models, LossConfigSet):
    
    ### Loss-related parameters
    Capacity_Z = LossConfigSet['Capacity_Z']
    Capacity_Fc = LossConfigSet['Capacity_Fc']
    

    
    ###-------------------------------- Model structuring -------------------------------------------- ###
    ## Model core parts
    EncModel,FeatExtModel,FeatGenModel,ReconModel,FacDiscModel =  Models
    EncInp =EncModel.input
    InpZ = EncModel.output[2]
    InpFCCommon = EncModel.output[1][:, :2]
    InpFCEach = EncModel.output[1][:, 2:]


    ## Batch split 
    BatchSize = tf.shape(EncInp)[0]
    HalfBatchIdx1 = tf.range(BatchSize//2)
    HalfBatchIdx2 = tf.range(BatchSize//2, BatchSize)
    Z_D1, Z_D2 = SplitBatch(InpZ, HalfBatchIdx1, HalfBatchIdx2, mode='Both')


    ## Each output of each model
    FeatExtOut = FeatExtModel(EncModel.output[:2])
    FeatGenOut = FeatGenModel([InpFCCommon, InpFCEach, InpZ])
    ReconExtOut = ReconModel(FeatExtOut)
    ReconGenOut = ReconModel(FeatGenOut)
    FacDiscOut_D1 =  FacDiscModel(Z_D1)

    ### Define the total model
    SigRepModel = Model(EncInp, [FacDiscOut_D1, FeatGenOut, ReconExtOut, ReconGenOut])

    
    
    ###-------------------------------- Weights for losses -------------------------------------------- ###
    ### Weight controller; Apply beta and capacity 
    Beta_Z = Lossweight(name='Beta_Z', InitVal=0.01)(SigRepModel.input)
    Beta_Fc = Lossweight(name='Beta_Fc', InitVal=0.01)(SigRepModel.input)
    Beta_TC = Lossweight(name='Beta_TC', InitVal=0.01)(SigRepModel.input)
    Beta_Orig = Lossweight(name='Beta_Orig', InitVal=1.)(SigRepModel.input)
    Beta_Feat = Lossweight(name='Beta_Feat', InitVal=1.)(SigRepModel.input)



    ###-------------------------------- Common losses -------------------------------------------- ###
    ### Adding the RecLoss; 
    MSE = tf.keras.losses.MeanSquaredError()
    ReconExtOut_D1 = SplitBatch(ReconExtOut, HalfBatchIdx1, HalfBatchIdx2, mode='D1')
    EncInp_D1 = SplitBatch(EncInp, HalfBatchIdx1, HalfBatchIdx2, mode='D1')
    ReconOut_ext = Beta_Orig * MSE(ReconExtOut_D1, EncInp_D1)

    SigRepModel.add_loss(ReconOut_ext)
    SigRepModel.add_metric(ReconOut_ext, 'OrigRecLoss')
    print('OrigRecLoss added')



    ### Adding the FeatRecLoss; It allows connection between the extractor and generator
    FeatGenOut_D1 = SplitBatch(tf.concat(FeatGenOut, axis=-1), HalfBatchIdx1, HalfBatchIdx2, mode='D1')
    FeatExtOut_D1 = SplitBatch(tf.concat(FeatExtOut, axis=-1), HalfBatchIdx1, HalfBatchIdx2, mode='D1')
    FeatRecLoss= Beta_Feat * MSE(FeatGenOut_D1, FeatExtOut_D1)

    SigRepModel.add_loss(FeatRecLoss)
    SigRepModel.add_metric(FeatRecLoss, 'FeatRecLoss')
    print('FeatRecLoss added')



    ### KL Divergence for p(Z) vs q(Z)
    Z_Mu, Z_Log_Sigma, Zs = SigRepModel.get_layer('Z_Mu').output, SigRepModel.get_layer('Z_Log_Sigma').output, SigRepModel.get_layer('Zs').output
    Z_Mu_D1 = SplitBatch(Z_Mu, HalfBatchIdx1, HalfBatchIdx2, mode='D1')
    Z_Log_Sigma_D1 = SplitBatch(Z_Log_Sigma, HalfBatchIdx1, HalfBatchIdx2, mode='D1')

    kl_Loss_Z = 0.5 * tf.reduce_sum( Z_Mu_D1**2  +  tf.exp(Z_Log_Sigma_D1)- Z_Log_Sigma_D1-1, axis=1)    
    kl_Loss_Z = tf.reduce_mean(kl_Loss_Z )
    kl_Loss_Z = Beta_Z * tf.abs(kl_Loss_Z - Capacity_Z)
    SigRepModel.add_loss(kl_Loss_Z )
    SigRepModel.add_metric(kl_Loss_Z , 'kl_Loss_Z')
    print('SKZ loss selected')


    #### ### Total Correlation # KL(q(z)||prod_j q(z_j)); log(p_true/p_false) = logit_true - logit_false
    kl_Loss_TC = tf.reduce_mean(FacDiscOut_D1[:, 0] - FacDiscOut_D1[:, 1])
    kl_Loss_TC = Beta_TC * kl_Loss_TC
    SigRepModel.add_loss(kl_Loss_TC )
    SigRepModel.add_metric(kl_Loss_TC , 'kl_Loss_TC')
    print('kl_Loss_TC added')

    
    # Discriminator Loss
    Batch2_Size = tf.shape(Z_D2)[0]
    PermZ_D2 = tf.concat([tf.nn.embedding_lookup(Z_D2[:, i][:,None], tf.random.shuffle(tf.range(Batch2_Size))) for i in range(Z_D1.shape[-1])], axis=1)
    PermZ_D2 = tf.stop_gradient(PermZ_D2) # 
    FacDiscOut_D2 = FacDiscModel(PermZ_D2)

    Ones = tf.ones_like(HalfBatchIdx1)[:,None]
    Zeros = tf.zeros_like(HalfBatchIdx2)[:,None]
    CCE = tf.keras.losses.MeanSquaredError()
    kl_Loss_DTC = 0.5 * (CCE(Zeros, FacDiscOut_D1) + CCE(Ones, FacDiscOut_D2))
    SigRepModel.add_loss(kl_Loss_DTC )
    SigRepModel.add_metric(kl_Loss_DTC , 'kl_Loss_DTC')
    print('kl_Loss_DTC added')



    ###-------------------------------- Specific losses -------------------------------------------- ###
    ### KL Divergence for p(FC) vs q(FC)
    BernP = 0.5 # hyperparameter
    FC_Mu = SigRepModel.get_layer('FC_Mu').output 
    FC_Mu_D1 = SplitBatch(FC_Mu, HalfBatchIdx1, HalfBatchIdx2, mode='D1')
    kl_Loss_FC = FC_Mu_D1*(tf.math.log(FC_Mu_D1) - tf.math.log(BernP)) + (1-FC_Mu_D1)*(tf.math.log(1-FC_Mu_D1) - tf.math.log(1-BernP))
    kl_Loss_FC = tf.reduce_mean(kl_Loss_FC )
    kl_Loss_FC = Beta_Fc * tf.abs(kl_Loss_FC - Capacity_Fc)
    SigRepModel.add_loss(kl_Loss_FC )
    SigRepModel.add_metric(kl_Loss_FC , 'kl_Loss_FC')
    print('kl_Loss_FC added')


        
    return SigRepModel