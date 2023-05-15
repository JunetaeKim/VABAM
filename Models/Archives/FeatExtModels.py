import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, GRU, Dense, Masking, Reshape, Flatten, RepeatVector, TimeDistributed, Bidirectional, Activation, GaussianNoise, Lambda
from tensorflow.keras import Model

def MaskingGen ( InpRegul, MaskingRate = 0.025, GaussStd2 = 0.1):
    ## Masking vector generation 1 vs 0
    NBatch = tf.shape(InpRegul)[0]
    
    MaskIDX = tf.random.shuffle(tf.range(NBatch * InpRegul.shape[1] ))
    CutIDX = tf.cast(  tf.cast(tf.shape(MaskIDX)[0], dtype=tf.float32) * (1-MaskingRate), dtype=tf.int32 )
    MaskIDX = tf.cast(MaskIDX < CutIDX, dtype=tf.float32)
    MaskVec = tf.reshape(MaskIDX, (NBatch, -1))[:,:,None]
    
    ## Noise vector generation 1 vs 0
    RevMaskIDX = MaskIDX == 0
    RevMaskIDX = tf.cast(RevMaskIDX, dtype=tf.float32)
    NoisVec = RevMaskIDX * tf.random.normal(tf.shape(RevMaskIDX), stddev=GaussStd2)
    NoisVec = tf.reshape(NoisVec, (NBatch, -1))[:,:,None]
    return MaskVec, NoisVec

def GenLowFilter (LF, BL, N = 401):
    nVec = np.arange(N)
    KaiserBL = tf.signal.kaiser_window(N, beta=BL)

    # A low-pass filter
    X = 2 * LF * (nVec - (N - 1) / 2)
    X = tf.maximum(X , 1e-7)
    LPF = tf.sin(np.pi*X)/(np.pi*X)
    LPF *= KaiserBL
    LPF /= tf.reduce_sum(LPF, axis=-1, keepdims=True)
    
    return LPF[:,None] 


def GenHighFilter (HF, BH, N = 401):
    nVec = np.arange(N)
    KaiserBH = tf.signal.kaiser_window(N, beta=BH)

    # A high-pass filter
    Y = 2 * HF * (nVec - (N - 1) / 2)
    Y = tf.maximum(Y , 1e-7)
    HPF = tf.sin(np.pi*Y)/(np.pi*Y)
    HPF *= KaiserBH
    HPF /= tf.reduce_sum(HPF, axis=-1, keepdims=True)
    HPF = -HPF

    ## HPF[(N - 1) // 2] += 1
    Mask = np.zeros(HPF.shape[1])
    Mask[(N - 1) // 2] += 1
    Mask = tf.constant(Mask, dtype=tf.float32)
    HPF = HPF + Mask
    
    return HPF[:,None] 

def ReName (layer, name):
    return Lambda(lambda x: x, name=name)(layer)



def EncoderModel_FX(SigDim, Type = '', LatDim= 2, MaskingRate = 0.025,GaussStd1 = 0.002, GaussStd2 = 0.1, training = False):
    
    InpL = Input(shape=(SigDim,))
    InpFrame = tf.signal.frame(InpL, 100, 100)

    if training:
        InpRegul = GaussianNoise(stddev=GaussStd1)(InpFrame, training=training)
        MaskVec, NoisVec = MaskingGen(InpRegul, MaskingRate, GaussStd2)
        EncInp = Masking(mask_value=0.)(InpRegul * MaskVec )
        EncOut = InpRegul + NoisVec
    else:
        EncInp, EncOut = InpFrame, InpFrame

    EncGRUOut = Bidirectional(GRU(50, return_sequences=True))(EncInp)
    EncGRUOut = Bidirectional(GRU(20, return_sequences=False))(EncGRUOut)

    Zs = Dense(20, activation='relu')(EncGRUOut)
    Zs = Dense(10, activation='relu')(Zs)
    Z_Mean = Dense(LatDim, activation='linear')(Zs) 
    Z_Log_Sigma = Dense(LatDim, activation='softplus', name='Z_Log_Sigma_'+Type)(Zs)

    # Reparameterization Trick for sampling 
    Epsilon = tf.random.normal(shape=(tf.shape(Z_Mean)[0], Z_Mean.shape[1]), mean=0., stddev=0.1)
    
    if training==False:
        Epsilon = Epsilon * 0
        
    Z_Mean = Z_Mean + tf.exp(0.5 * Z_Log_Sigma) * Epsilon
    Z_Mean = ReName(Z_Mean,'Z_Mean_'+Type)
    
    EncModel = Model(InpL, [Flatten()(EncOut), Z_Mean, Z_Log_Sigma], name='EncoderModel')
    
    return EncModel


def FeatExtractModel(SigDim, LatDim= 2):
    
    EncReInp = Input(shape=(SigDim,))
    InpZ = Input(shape=(LatDim,), name='InpZmean')
    
    FiltPar = Dense(4, activation='relu')(InpZ)
    FiltPar = Dense(8, activation='relu')(FiltPar)
    FiltPar = Dense(4*3, activation='linear')(FiltPar) # Num. of parameters for filtering : 4, Num. of filtering : 3 

    LHFs = Activation('sigmoid')(FiltPar[:, :(4*3)//2])
    LHFs = Reshape((-1,2), name='LHFs')(LHFs)

    LHBs = Activation('softplus')(FiltPar[:, (4*3)//2:])
    LHBs = Reshape((-1,2), name='LHBs')(LHBs)

    
    ### Filtering level 1 -------------------------------------------------------------------
    ## Filter generation
    HPF_11 = GenHighFilter(LHFs[:, 0, 0:1], LHBs[:, 0, 0:1], N=301)
    LPF_11 = GenLowFilter(LHFs[:, 0, 1:2],  LHBs[:, 0, 1:2],  N=201)

    ## Perform signal filtering level 1
    InpFrame_11 =  tf.signal.frame(EncReInp, HPF_11.shape[-1], 1)
    HpSig_11 = tf.reduce_sum(InpFrame_11*HPF_11[:,:,::-1], axis=-1, keepdims=True)
    HpSig_11 = ReName(HpSig_11, 'HpSig_11')
    InpFrame_11 =  tf.signal.frame(EncReInp, LPF_11.shape[-1], 1)
    LpSig_11 = tf.reduce_sum(InpFrame_11*LPF_11[:,:,::-1], axis=-1, keepdims=True)
    LpSig_11 = ReName(LpSig_11, 'LpSig_11')


    ### Filtering level 21 (from HpSig_11) -------------------------------------------------------------------
    ## Filter generation
    HPF_21 = GenHighFilter(LHFs[:, 1, 0:1], LHBs[:, 1, 0:1], N=401)
    LPF_21 = GenLowFilter(LHFs[:, 1, 1:2],  LHBs[:, 1, 1:2],  N=301)

    ## Perform signal filtering level 2
    InpFrame_21 =  tf.signal.frame(HpSig_11[:,:,0], HPF_21.shape[-1], 1)
    HpSig_21 = tf.reduce_sum(InpFrame_21*HPF_21[:,:,::-1], axis=-1, keepdims=True)
    HpSig_21 = ReName(HpSig_21, 'HpSig_21')
    InpFrame_21 =  tf.signal.frame(HpSig_11[:,:,0], LPF_21.shape[-1], 1)
    LpSig_21 = tf.reduce_sum(InpFrame_21*LPF_21[:,:,::-1], axis=-1, keepdims=True)
    LpSig_21 = ReName(LpSig_21, 'LpSig_21')

    

    ### Filtering level 22 (from LpSig_11) -------------------------------------------------------------------
    ## Filter generation
    HPF_22 = GenHighFilter(LHFs[:, 2, 0:1], LHBs[:, 2, 0:1], N=301)
    LPF_22 = GenLowFilter(LHFs[:, 2, 1:2],  LHBs[:, 2, 1:2],  N=201)

    ## Perform signal filtering level 2
    InpFrame_22 =  tf.signal.frame(LpSig_11[:,:,0], HPF_22.shape[-1], 1)
    HpSig_22 = tf.reduce_sum(InpFrame_22*HPF_22[:,:,::-1], axis=-1, keepdims=True)
    HpSig_22 = ReName(HpSig_22, 'HpSig_22')

    InpFrame_22 =  tf.signal.frame(LpSig_11[:,:,0], LPF_22.shape[-1], 1)
    LpSig_22 = tf.reduce_sum(InpFrame_22*LPF_22[:,:,::-1], axis=-1, keepdims=True)
    LpSig_22 = ReName(LpSig_22, 'LpSig_22')
    
    FeatExM = Model([EncReInp, InpZ], [Flatten()(HpSig_21), Flatten()(LpSig_21), Flatten()(HpSig_22), Flatten()(LpSig_22)], name='FeatExtractModel')
    
    return FeatExM




def DecoderModel(SigDim):
    
    HpSig_21 = Input(shape=(300,))
    LpSig_21 = Input(shape=(400,))
    HpSig_22 = Input(shape=(500,))
    LpSig_22 = Input(shape=(600,))

    ## GRU NET -------------------------------------------------------------------
    Dec_HpSig_21 = Reshape((-1, 100))(HpSig_21)
    Dec_LpSig_21 = Reshape((-1, 100))(LpSig_21)
    Dec_HpSig_22 = Reshape((-1, 100))(HpSig_22)
    Dec_LpSig_22 = Reshape((-1, 100))(LpSig_22)

    Dec_HpSig_21 = Bidirectional(GRU(5))(Dec_HpSig_21)
    Dec_LpSig_21 = Bidirectional(GRU(5))(Dec_LpSig_21)
    Dec_HpSig_22 = Bidirectional(GRU(5))(Dec_HpSig_22)
    Dec_LpSig_22 = Bidirectional(GRU(5))(Dec_LpSig_22)

    Decoder = tf.concat([ Dec_HpSig_21, Dec_LpSig_21, Dec_HpSig_22, Dec_LpSig_22], axis=1)
    Decoder = RepeatVector((SigDim//100) )(Decoder)
    Decoder = Bidirectional(GRU(50, return_sequences=True))(Decoder)
    Decoder = TimeDistributed(Dense(100))(Decoder)
    #Decoder = Bidirectional(GRU(50, return_sequences=True))(Decoder)
    DecOut = Dense(100, activation='sigmoid')(Decoder)
    DecOut = Reshape((SigDim,),name='DecOut')(DecOut)
    DecModel = Model([HpSig_21, LpSig_21, HpSig_22, LpSig_22], DecOut, name='DecoderModel')
    
    return DecModel


def VaeModel (EncModel, FeExModel,  DecModel):
    
    FeExOut = FeExModel(EncModel.output[:2])
    VaeSFOut = DecModel(FeExOut)
    VaeSFOut = Flatten(name='Out')(VaeSFOut)
    VaeSFModel = Model(EncModel.input, VaeSFOut, name='VaeSFModel')
    
    return VaeSFModel