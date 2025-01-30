import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64') # Set the default float type for TensorFlow to float64
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, GRU, Dense, Masking, Reshape, Flatten, RepeatVector, Bidirectional, Activation, GaussianNoise
from tensorflow.keras import Model
from Utilities.Utilities import ReName


def MaskingGen ( InpRegul, MaskingRate, MaskStd):
    ## Masking vector generation 1 vs 0
    NBatch = tf.shape(InpRegul)[0]
    
    MaskIDX = tf.random.shuffle(tf.range(NBatch * InpRegul.shape[1] ))
    CutIDX = tf.cast(  tf.cast(tf.shape(MaskIDX)[0], dtype=tf.float64) * (1-MaskingRate), dtype=tf.int32 )
    MaskIDX = tf.cast(MaskIDX < CutIDX, dtype=tf.float64)
    MaskVec = tf.reshape(MaskIDX, (NBatch, -1))[:,:,None]
    
    # Generate noise vector for unmasked positions (where mask is 0)
    RevMaskIDX = MaskIDX == 0
    RevMaskIDX = tf.cast(RevMaskIDX, dtype=tf.float64)
    NoisVec = RevMaskIDX * tf.random.normal(tf.shape(RevMaskIDX), stddev=MaskStd, dtype=tf.float64)
    NoisVec = tf.reshape(NoisVec, (NBatch, -1))[:,:,None]
    return MaskVec, NoisVec

def GenLowFilter (LF, N, Decay=0.):
    nVec = np.arange(N, dtype=np.float64)
    Window = tf.signal.hamming_window(N) # Window shape = [N]

    # A low-pass filter
    X = 2 * LF * (nVec - (N - 1) / 2)
    X = tf.where(X == 0, tf.constant(1e-40, dtype=tf.float64), X)
    LPF = tf.sin(np.pi*X)/(np.pi*X)
    LPF *= tf.cast(Window, tf.float64)
    LPF /= tf.reduce_sum(LPF, axis=-1, keepdims=True)
    
    # Freq cutoff Decay effect
    if Decay != 0:
        LPF *= tf.exp(-Decay * nVec) 
        #LPF *= tf.exp(-LF * Decay * nVec) 
            
    return LPF[:,None]  

def GenHighFilter (HF, N, Decay=0.):
    nVec = np.arange(N, dtype=np.float64)
    Window = tf.signal.hamming_window(N)

    # A high-pass filter
    Y = 2 * HF * (nVec - (N - 1) / 2)
    Y = tf.where(Y == 0, tf.constant(1e-40, dtype=tf.float64), Y)
    HPF = tf.sin(np.pi*Y)/(np.pi*Y)
    HPF *= tf.cast(Window, tf.float64)
    HPF /= tf.reduce_sum(HPF, axis=-1, keepdims=True)
    HPF = -HPF

    ## To make HPF[(N - 1) // 2] += 1
    # Add center spike for high-pass characteristic
    Mask = np.zeros(HPF.shape[1])
    Mask[(N - 1) // 2] += 1
    Mask = tf.constant(Mask, dtype=tf.float64)
    HPF = HPF + Mask
    
    # Freq cutoff Decay effect
    if Decay != 0:
        HPF *= tf.exp(-Decay * nVec) 
        #HPF *= tf.exp(-HF * Decay * nVec) 
    return HPF[:,None] 



## --------------------------------------------------    Models   ------------------------------------------------------------
## --------------------------------------------------   Encoder  -------------------------------------------------------------
def Encoder(SigDim, SlidingSize = 50, LatDim= 2, Depth=2, Type = '', MaskingRate = 0., NoiseStd = 0., MaskStd = 0., ReparaStd = 0.1 , Reparam = False, FcLimit=0.05):

    NFilters = sum(2 ** i for i in range(Depth)) * 2

    InpL = Input(shape=(SigDim,), name='Inp_Enc')
    InpFrame = tf.signal.frame(InpL, SlidingSize, SlidingSize)

    if Reparam: 
        '''MaskingGen and InpRegul parts have been skipped in the research. 
        These can be activated for the purpose of enhancing model prediction generalization by adding noise. '''
        InpRegul = GaussianNoise(stddev=NoiseStd)(InpFrame, training=Reparam)
        MaskVec, NoisVec = MaskingGen(InpRegul, MaskingRate, MaskStd)
        EncInp = Masking(mask_value=0.)(InpRegul * MaskVec )
        EncOut = InpRegul + NoisVec
    else:
        EncInp, EncOut = InpFrame, InpFrame

    #Encoder = Dense(50, activation='relu')(InpFrame)
    Encoder = Bidirectional(GRU(25, return_sequences=True))(InpFrame)
    Encoder = Bidirectional(GRU(25, return_sequences=True))(Encoder)
    Encoder = Bidirectional(GRU(75, return_sequences=False))(Encoder)
    Encoder = Dense(100, activation='relu')(Encoder)
    Encoder = Dense(70, activation='relu')(Encoder)
    Encoder = Dense(50, activation='relu')(Encoder)

    Z_Mu = Dense(LatDim, activation='linear', name='Z_Mu')(Encoder)
    Z_Log_Sigma = Dense(LatDim, activation='softplus')(Encoder)
    Z_Log_Sigma = ReName(Z_Log_Sigma,'Z_Log_Sigma'+Type)

    
    # Reparameterization Trick for sampling from Guassian distribution
    Epsilon_z = tf.random.normal(shape=(tf.shape(Z_Mu)[0], Z_Mu.shape[1]), mean=0., stddev=ReparaStd)

    if Reparam==False:
        Epsilon_z = Epsilon_z * 0

    Zs = Z_Mu + tf.exp(0.5 * Z_Log_Sigma) * tf.cast(Epsilon_z, tf.float64)
    Zs = ReName(Zs,'Zs'+Type)
    
    FC_Mu =  Dense(NFilters, activation='relu')(Encoder)
    FC_Mu =  Dense(NFilters, activation='sigmoid')(FC_Mu)
    FC_Mu = tf.clip_by_value(FC_Mu, 1e-7, 1-1e-7)
    FC_Mu = ReName(FC_Mu,'FC_Mu')
    
    # Reparameterization Trick for sampling from Uniformly distribution; ϵ∼U(0,1) 
    Epsilon_fc = tf.random.uniform(shape=(tf.shape(FC_Mu)[0], FC_Mu.shape[1]))
    Epsilon_fc = tf.clip_by_value(tf.cast(Epsilon_fc, tf.float64), 1e-30, 1-1e-30)

    LogEps = tf.math.log(Epsilon_fc)
    LogNegEps = tf.math.log(1 - Epsilon_fc)
    
    LogTheta = tf.math.log(FC_Mu)
    LogNegTheta = tf.math.log(1-FC_Mu)
    
    if Reparam==True:
        FCs = tf.math.sigmoid(LogEps - LogNegEps + LogTheta - LogNegTheta) * 1. + FC_Mu * 0.
    else:
        FCs = tf.math.sigmoid(LogEps - LogNegEps + LogTheta - LogNegTheta) * 0. + FC_Mu * 1.
    
    FCs = FCs * FcLimit
    FCs = tf.clip_by_value(FCs, 1e-7, FcLimit-1e-7)
    FCs = ReName(FCs, 'FCs')
    
    return Model(InpL, [Flatten(name='SigOut')(EncOut), FCs, Zs], name='EncModel') 


## --------------------------------------------------   FeatExtractor  -------------------------------------------------------------

def Filtering(signal, H_F, L_F, DecayH=0, DecayL=0, FiltLen=16):
    # Generate filters for this level
    To_H = GenHighFilter(H_F, N=FiltLen, Decay=DecayH)
    To_L = GenLowFilter(L_F, N=FiltLen, Decay=DecayL)
    
    # Perform filtering for high-pass
    InpFrame_H = tf.signal.frame(signal, To_H.shape[-1], 1)
    Sig_H = tf.reduce_sum(InpFrame_H * To_H[:, :, ::-1], axis=-1, keepdims=True)
    
    # Perform filtering for low-pass
    InpFrame_L = tf.signal.frame(signal, To_L.shape[-1], 1)
    Sig_L = tf.reduce_sum(InpFrame_L * To_L[:, :, ::-1], axis=-1, keepdims=True)
    
    return Flatten()(Sig_H), Flatten()(Sig_L)


def Recursion(signal, FCval, DecayH=0, DecayL=0, FiltLen=16):

    sig_res = []

    for i in range(signal.shape[1]):
        H_F, L_F = FCval[:, i, 0:1], FCval[:, i, 1:2]
        sig_res.append(tf.stack(Filtering(signal[:, i], H_F, L_F, FiltLen=FiltLen), axis=1))
    sig_res = tf.concat(sig_res, axis=1)

    if i+1 == FCval.shape[1]: # end condition
        #print(i+1, FCval.shape[1])
        return sig_res
    else:                   # recursive condition
        #print(i+1, FCval.shape[1])
        return Recursion(sig_res, FCval[:, i+1:], FiltLen=FiltLen)


def FeatExtractor(SigDim, LatDim= 2, CompSize = 600, Depth=2, DecayH = 0. , DecayL = 0. ):


    NFilters = sum(2 ** i for i in range(Depth)) * 2
    KernelSize = (SigDim - CompSize)//Depth + 1
    
    EncReInp = Input(shape=(SigDim,), name='Inp_EncRe')
    FCs = Input(shape=(NFilters,), name='Inp_FCs')
    FCs_exp = Reshape((-1,2))(FCs)
    
    ExtractedFeats = Recursion(EncReInp[:,None], FCs_exp, DecayH=0, DecayL=0, FiltLen=KernelSize)
    
    return Model([EncReInp, FCs], tf.unstack(ExtractedFeats, axis=1), name='FeatExtModel') 


## --------------------------------------------------   FeatGenerator  -------------------------------------------------------------
def BlockFeatGen(InpZ, FCCommon, FCUnique, SigDim, CompSize, SlidingSize= 50):

    Dec_Sig = Dense(50, activation='relu')(tf.concat([InpZ, FCCommon, FCUnique], axis=-1))
    Dec_Sig = Dense(50, activation='relu')(Dec_Sig)
    Dec_Sig = Dense(50, activation='relu')(Dec_Sig)
    Dec_Sig = Dense(75, activation='relu')(Dec_Sig)

    Dec_Sig = RepeatVector(SigDim//SlidingSize )(Dec_Sig)
    Dec_Sig = Bidirectional(GRU(10, return_sequences=True))(Dec_Sig)
    Dec_Sig = Bidirectional(GRU(10, return_sequences=True))(Dec_Sig)
    Dec_Sig = Bidirectional(GRU(10, return_sequences=True))(Dec_Sig)
    Dec_Sig = Dense(CompSize//Dec_Sig.shape[1],'tanh')(Dec_Sig)
    
    return Flatten()(Dec_Sig)


def FeatGenerator (SigDim, CompSize, NCommonFC, NeachFC, LatDim= 2, SlidingSize= 50):
    
    InpZ = Input(shape=(LatDim,), name='Inp_Z')
    FCCommon = Input(shape=(NCommonFC,), name='Inp_FCCommon')
    FCEach = Input(shape=(NeachFC,), name='Inp_FCEach')
    FeatGenList = [BlockFeatGen(InpZ, FCCommon, FCEach[:, i:i+1], SigDim, CompSize, SlidingSize=SlidingSize) for i in range(NeachFC)]

    return  Model([FCCommon, FCEach, InpZ], FeatGenList, name='FeatGenModel')



## --------------------------------------------------   Reconstructor  -------------------------------------------------------------
def Reconstructor(SigDim , NeachFC, SlidingSize = 50, CompSize=600 ):
    
    SigList = [Input(shape=(CompSize,), name='Inp_Sig_'+str(i)) for i in range(NeachFC)]
    DecSigList = [Bidirectional(GRU(20))(Reshape((-1, SlidingSize))(Sig)) for Sig in SigList]
    
    Decoder = tf.concat(DecSigList, axis=1)
    Decoder = Dense(Decoder.shape[-1], activation='relu')(Decoder)
    Decoder = Dense(Decoder.shape[-1], activation='relu')(Decoder)
    Decoder = Dense(Decoder.shape[-1], activation='relu')(Decoder)
    Decoder = Dense(Decoder.shape[-1], activation='relu')(Decoder)
    Decoder = RepeatVector((SigDim//SlidingSize) )(Decoder)
    Decoder = Bidirectional(GRU(25, return_sequences=True))(Decoder)
    Decoder = Bidirectional(GRU(25, return_sequences=True))(Decoder)
    Decoder = Bidirectional(GRU(25, return_sequences=True))(Decoder)
    DecOut = Dense(SlidingSize,'sigmoid')(Decoder)
    DecOut = Reshape((SigDim,),name='Out')(DecOut)
    
    return Model(SigList, DecOut, name='ReconModel')



