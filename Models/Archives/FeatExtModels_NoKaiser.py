import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, GRU, Dense, Masking, Reshape, Flatten, RepeatVector, TimeDistributed, Bidirectional, Activation, GaussianNoise, Lambda
from tensorflow.keras import Model



def MaskingGen ( InpRegul, MaskingRate = 0.025, MaskStd = 0.1):
    ## Masking vector generation 1 vs 0
    NBatch = tf.shape(InpRegul)[0]
    
    MaskIDX = tf.random.shuffle(tf.range(NBatch * InpRegul.shape[1] ))
    CutIDX = tf.cast(  tf.cast(tf.shape(MaskIDX)[0], dtype=tf.float32) * (1-MaskingRate), dtype=tf.int32 )
    MaskIDX = tf.cast(MaskIDX < CutIDX, dtype=tf.float32)
    MaskVec = tf.reshape(MaskIDX, (NBatch, -1))[:,:,None]
    
    ## Noise vector generation 1 vs 0; Enocder 에서는 masking {0 vs 1}을 통해 BP가 되지 않도록 제어; Filtering (feature extracting) part에서는 encoder에서 masking 된 부분에 noise를 추가해줌
    RevMaskIDX = MaskIDX == 0
    RevMaskIDX = tf.cast(RevMaskIDX, dtype=tf.float32)
    NoisVec = RevMaskIDX * tf.random.normal(tf.shape(RevMaskIDX), stddev=MaskStd)
    NoisVec = tf.reshape(NoisVec, (NBatch, -1))[:,:,None]
    return MaskVec, NoisVec

def GenLowFilter (LF, N = 401):
    nVec = np.arange(N)
    Window = tf.signal.hamming_window(N)

    # A low-pass filter
    X = 2 * LF * (nVec - (N - 1) / 2)
    X = tf.maximum(X , 1e-7)
    LPF = tf.sin(np.pi*X)/(np.pi*X)
    LPF *= Window
    LPF /= tf.reduce_sum(LPF, axis=-1, keepdims=True)
    
    return LPF[:,None] 


def GenHighFilter (HF, N = 401):
    nVec = np.arange(N)
    Window = tf.signal.hamming_window(N)

    # A high-pass filter
    Y = 2 * HF * (nVec - (N - 1) / 2)
    Y = tf.maximum(Y , 1e-7)
    HPF = tf.sin(np.pi*Y)/(np.pi*Y)
    HPF *= Window
    HPF /= tf.reduce_sum(HPF, axis=-1, keepdims=True)
    HPF = -HPF

    ## HPF[(N - 1) // 2] += 1
    Mask = np.zeros(HPF.shape[1])
    Mask[(N - 1) // 2] += 1
    Mask = tf.constant(Mask, dtype=tf.float32)
    HPF = HPF + Mask
    
    return HPF[:,None] 


def ParaFilters (layer):
    
    layer = Dense(2, activation='linear')(layer)
    
    return Activation('sigmoid')(layer[:, 0:1])


def ReName (layer, name):
    return Lambda(lambda x: x, name=name)(layer)




## ---------------------------------------------------------------------------------------------------------------------------------------
def EncoderModel (SigDim, LatDim= 2, Type = '', MaskingRate = 0.025, NoiseStd = 0.002, MaskStd = 0.1, ReparaStd = 0.1 ,training = False):
    InpL = Input(shape=(SigDim,))
    InpFrame = tf.signal.frame(InpL, 100, 100)

    if training:
        InpRegul = GaussianNoise(stddev=NoiseStd)(InpFrame, training=training)
        MaskVec, NoisVec = MaskingGen(InpRegul, MaskingRate, MaskStd)
        EncInp = Masking(mask_value=0.)(InpRegul * MaskVec )
        EncOut = InpRegul + NoisVec
    else:
        EncInp, EncOut = InpFrame, InpFrame

    EncGRUOut = Bidirectional(GRU(50, return_sequences=True))(EncInp)
    EncGRUOut = Bidirectional(GRU(20, return_sequences=False))(EncGRUOut)

    Zs_ft = Dense(20, activation='softplus')(EncGRUOut)
    Zs_ft = Dense(10, activation='softplus')(Zs_ft)


    Z_Mean_ft = tf.concat([Dense(LatDim, activation='linear')(Zs_ft)  for i in range(6)], axis=1)
    Z_Log_Sigma_ft = tf.concat([Dense(LatDim, activation='softplus')(Zs_ft)  for i in range(6)], axis=1)
    Z_Log_Sigma_ft = ReName(Z_Log_Sigma_ft,'Z_Log_Sigma_'+Type)

    # Reparameterization Trick for sampling 
    Epsilon = tf.random.normal(shape=(tf.shape(Z_Mean_ft)[0], Z_Mean_ft.shape[1]), mean=0., stddev=ReparaStd)

    if training==False:
        Epsilon = Epsilon * 0

    Z_Mean_ft = Z_Mean_ft + tf.exp(0.5 * Z_Log_Sigma_ft) * Epsilon
    Z_Mean_ft = ReName(Z_Mean_ft,'Z_Mean_'+Type)

    EncModel = Model(InpL, [Flatten()(EncOut), Z_Mean_ft, Z_Log_Sigma_ft], name='EncoderModel')
    
    return EncModel



## ---------------------------------------------------------------------------------------------------------------------------------------
def FeatExtractModel(LatDim= 2, FiltLenList = [301, 301, 301, 301, 301, 301]):
    
    EncReInp = Input(shape=(None,))
    InpZ = Input(shape=(LatDim*6,), name='InpZmean')

    Filt_H, Filt_L, Filt_HH, Filt_HL, Filt_LH, Filt_LL  = tf.split(InpZ, 6, axis=1)

    H_F = ParaFilters(Filt_H)
    L_F = ParaFilters(Filt_L)
    HH_F = ParaFilters(Filt_HH)
    HL_F = ParaFilters(Filt_HL)
    LH_F = ParaFilters(Filt_LH)
    LL_F = ParaFilters(Filt_LL)


    ### Filtering level 1 -------------------------------------------------------------------
    ## Filter generation
    To_H = GenHighFilter(H_F,  N=FiltLenList[0])
    To_L = GenLowFilter(L_F, N=FiltLenList[1])

    ## Perform signal filtering level 1
    InpFrame =  tf.signal.frame(EncReInp, To_H.shape[-1], 1)
    Sig_H = tf.reduce_sum(InpFrame*To_H[:,:,::-1], axis=-1, keepdims=True)
    Sig_H = ReName(Sig_H, 'Sig_H')

    InpFrame =  tf.signal.frame(EncReInp, To_L.shape[-1], 1)
    Sig_L = tf.reduce_sum(InpFrame*To_L[:,:,::-1], axis=-1, keepdims=True)
    Sig_L = ReName(Sig_L, 'Sig_L')



    ### Filtering level HH and HL (from Sig_H) -------------------------------------------------------------------
    ## Filter generation
    To_HH = GenHighFilter(HH_F, N=FiltLenList[2])
    To_HL = GenLowFilter(HL_F, N=FiltLenList[3])

    ## Perform signal filtering level 2
    Frame_H =  tf.signal.frame(Sig_H[:,:,0], To_HH.shape[-1], 1)
    Sig_HH = tf.reduce_sum(Frame_H*To_HH[:,:,::-1], axis=-1, keepdims=True)
    Sig_HH = ReName(Sig_HH, 'Sig_HH')

    Frame_H =  tf.signal.frame(Sig_H[:,:,0], To_HL.shape[-1], 1)
    Sig_HL = tf.reduce_sum(Frame_H*To_HL[:,:,::-1], axis=-1, keepdims=True)
    Sig_HL = ReName(Sig_HL, 'Sig_HL')


    ### Filtering level LH and LL (from Sig_L) -------------------------------------------------------------------
    ## Filter generation
    To_LH = GenHighFilter(LH_F,  N=FiltLenList[4])
    To_LL = GenLowFilter(LL_F,  N=FiltLenList[5])

    ## Perform signal filtering level 2
    Frame_L =  tf.signal.frame(Sig_L[:,:,0], To_LH.shape[-1], 1)
    Sig_LH = tf.reduce_sum(Frame_L*To_LH[:,:,::-1], axis=-1, keepdims=True)
    Sig_LH = ReName(Sig_LH, 'Sig_LH')

    Frame_L =  tf.signal.frame(Sig_L[:,:,0], To_LL.shape[-1], 1)
    Sig_LL = tf.reduce_sum(Frame_L*To_LL[:,:,::-1], axis=-1, keepdims=True)
    Sig_LL = ReName(Sig_LL, 'Sig_LL')

    FeatExM = Model([EncReInp, InpZ], [Flatten()(Sig_HH), Flatten()(Sig_HL), Flatten()(Sig_LH), Flatten()(Sig_LL)], name='FeatExtractModel')
    

    return FeatExM


## ---------------------------------------------------------------------------------------------------------------------------------------
def DecoderModel(SigDim):

    Sig_HH = Input(shape=(None,))
    Sig_HL = Input(shape=(None,))
    Sig_LH = Input(shape=(None,))
    Sig_LL = Input(shape=(None,))

    ## GRU NET -------------------------------------------------------------------
    Dec_Sig_HH = Reshape((-1, 100))(Sig_HH)
    Dec_Sig_HL = Reshape((-1, 100))(Sig_HL)
    Dec_Sig_LH = Reshape((-1, 100))(Sig_LH)
    Dec_Sig_LL = Reshape((-1, 100))(Sig_LL)

    Dec_Sig_HH = Bidirectional(GRU(5), name='Dec_Sig_HH')(Dec_Sig_HH)
    Dec_Sig_HL = Bidirectional(GRU(5), name='Dec_Sig_HL')(Dec_Sig_HL)
    Dec_Sig_LH = Bidirectional(GRU(5), name='Dec_Sig_LH')(Dec_Sig_LH)
    Dec_Sig_LL = Bidirectional(GRU(5), name='Dec_Sig_LL')(Dec_Sig_LL)

    Decoder = tf.concat([ Dec_Sig_HH, Dec_Sig_HL, Dec_Sig_LH, Dec_Sig_LL], axis=1)
    Decoder = RepeatVector((SigDim//100) )(Decoder)
    Decoder = Bidirectional(GRU(50, return_sequences=True))(Decoder)
    Decoder = Dense(100, activation='softplus')(Decoder)
    #Decoder = Bidirectional(GRU(50, return_sequences=True))(Decoder)
    DecOut = Dense(100, activation='sigmoid')(Decoder)
    DecOut = Reshape((SigDim,),name='DecOut')(DecOut)
    DecModel = Model([Sig_HH, Sig_HL, Sig_LH, Sig_LL], DecOut, name='DecoderModel')
    
    return DecModel


## ---------------------------------------------------------------------------------------------------------------------------------------
def VaeModel (EncModel, FeExModel,  DecModel):
    
    FeExOut = FeExModel(EncModel.output[:2])
    VaeSFOut = DecModel(FeExOut)
    VaeSFOut = Reshape((-1,), name='Out')(VaeSFOut)
    VaeSFModel = Model(EncModel.input, VaeSFOut, name='VaeSFModel')
    
    return VaeSFModel