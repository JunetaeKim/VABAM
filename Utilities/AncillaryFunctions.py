import numpy as np
from scipy.stats import mode
import itertools
from tqdm import trange, tqdm

import tensorflow as tf
from tensorflow.keras import Model

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable



# For the dimensional Kullback-Leibler Divergence of the Z distribution
def LogNormalDensity(LatSamp, LatMean, LogSquaScale):
    Norm = tf.math.log(2. * tf.constant(np.pi))
    InvSigma = tf.math.exp(-LogSquaScale)
    MeanSampDiff = (LatSamp - LatMean)
    return -0.5 * (MeanSampDiff * MeanSampDiff * InvSigma + LogSquaScale + Norm)


# For Factor-VAE
def SplitBatch (Vec, HalfBatchIdx1, HalfBatchIdx2, mode='Both'):
    
    HalfBatch1 = tf.nn.embedding_lookup(Vec, HalfBatchIdx1)
    HalfBatch2 = tf.nn.embedding_lookup(Vec, HalfBatchIdx2)
    
    if mode=='Both':
        return  HalfBatch1, HalfBatch2
    elif mode=='D1':
        return  HalfBatch1
    elif mode=='D2':
        return  HalfBatch2
    

# Power spectral density 
def FFT_PSD (Data, ReducedAxis, MinFreq = 1, MaxFreq = 51):
    # Dimension check; this part operates with 3D tensors.
    # (Batch_size, N_sample, N_frequency)
    Data = Data[:,None] if len(Data.shape) < 3 else Data

    # Power Spectral Density
    HalfLen = Data.shape[-1]//2
    FFTRes = np.abs(np.fft.fft(Data, axis=-1)[..., :HalfLen])[..., MinFreq:MaxFreq]
    # (Batch_size, N_sample, N_frequency)
    PSD = (FFTRes**2)/Data.shape[-1]

    # Probability Density Function
    if ReducedAxis == 'All':
        AggPSD = np.mean(PSD, axis=(0,1))
        # (N_frequency,)
        AggPSPDF = AggPSD / np.sum(AggPSD, axis=(-1),keepdims=True)
    
    elif ReducedAxis =='Sample':
        AggPSD = np.mean(PSD, axis=(1))
        # (Batch_size, N_frequency)
        AggPSPDF = AggPSD / np.sum(AggPSD, axis=(-1),keepdims=True)
    
    elif ReducedAxis == 'None':
        # (Batch_size, N_sample, N_frequency)
        AggPSPDF = PSD / np.sum(PSD, axis=(-1),keepdims=True)    
        
    return AggPSPDF


# Permutation entropy given PSD over each generation
def ProbPermutation(Data, WindowSize=3):
    # Data shape: (Batch_size, N_frequency, N_sample)
    
    # Generate true permutation cases
    TruePerms = np.concatenate(list(itertools.permutations(np.arange(WindowSize)))).reshape(-1, WindowSize)

    # Get all permutation cases
    Data_Ext = tf.signal.frame(Data, frame_length=WindowSize, frame_step=1, axis=-1)
    PermsTable =  np.argsort(Data_Ext, axis=-1)

    CountPerms = 1- (TruePerms[None,None,None] == PermsTable[:,:,:, None])
    CountPerms = 1-np.sum(CountPerms, axis=-1).astype('bool')
    CountPerms = np.sum(CountPerms, axis=(2))
    
    # Data shape: (Batch_size, N_frequency, N_permutation_cases)
    ProbCountPerms = CountPerms / np.sum(CountPerms, axis=-1, keepdims=True)
    
    return np.maximum(ProbCountPerms, 1e-7)    



def MeanKLD(P,Q):
    return np.mean(np.sum(P*np.log(P/Q), axis=-1))



def Sampler (Data, SampModel,BatchSize=100, GPU=True):
    if GPU==False:
        with tf.device('/CPU:0'):
            PredVal = SampModel.predict(Data, batch_size=BatchSize, verbose=1)   
    else:
        PredVal = SampModel.predict(Data, batch_size=BatchSize, verbose=1)   

    return PredVal



def SamplingZ (Data, SampModel, NMiniBat, NGen, BatchSize = 1000, GPU=True, SampZType='GaussRptA'):
    
    '''
    Sampling Samp_Z 

    - Shape of UniqSamp_Z: (NMiniBat, LatDim)
    - UniqSamp_Z ~ N(Zμ|y, σ) for Type =='Model'
    - RandSamp_Z ~ N(0, ReparaStdZj) for Type =='Gauss'

    - Samp_Z is a 3D tensor expanded by repeating the first axis (i.e., 0) of UniqSamp_Z or RandSamp_Z by NGen times.
    - Shape of Samp_Z: (NMiniBat, NGen, LatDim) -> (NMiniBat*NGen, LatDim) for optimal use of GPU 
    
    - ModelRptA: The predicted values are repeated NGen times after the prediction. 
                 It is strongly recommended in cases where there are variations in the ancillary data inputs.  
    - ModelRptB: The data is repeated NGen times before the prediction.
                 It is strongly recommended when there is no ancillary data inputs or variations in the ancillary data.
    - GaussRptA: The data sampled from the Gaussian distribution is repeated NGen times.
                 It is recommended for both cases: when there are no ancillary data inputs and when there is ancillary data input.
                  
    '''
    
    # Sampling Samp_Z
    if SampZType =='ModelRptA': # Z ~ N(Zμ|y, σ);
        UniqSamp_Z = Sampler(Data, SampModel, GPU=GPU)
        Samp_Z =  np.broadcast_to(UniqSamp_Z[:, None], (NMiniBat, NGen, UniqSamp_Z.shape[-1])).reshape(-1, UniqSamp_Z.shape[-1])
    
    elif SampZType =='ModelRptB': # Z ~ N(Zμ|y, σ)
        DataRpt = np.repeat(Data, NGen, axis=0)
        Samp_Z = Sampler(DataRpt, SampModel, GPU=GPU)
        
    elif SampZType =='Gauss': # Z ~ N(0, ReparaStdZj)
        Samp_Z = np.random.normal(0, ReparaStdZj, (NMiniBat*NGen , SampModel.output.shape[-1]))
        
    elif SampZType =='GaussRptA': # Z ~ N(0, ReparaStdZj)
        UniqSamp_Z = np.random.normal(0, ReparaStdZj, (NMiniBat , SampModel.output.shape[-1]))
        Samp_Z = np.repeat(UniqSamp_Z, NGen, axis=0)
    
    return Samp_Z



def SamplingZj (Samp_Z, NMiniBat, NGen, LatDim, NSelZ, Axis=1):
    
    '''
     Sampling Samp_Zj 

    - Masking is applied to select Samp_Zj from Samp_Z 
      by assuming that the Samp_Z with indices other than j have a fixed mean value of '0' following a Gaussian distribution.

    - Shape of Samp_Zj: (NMiniBat, NGen, LatDim) -> (NMiniBat*NGen, LatDim) for optimal use of GPU 

    - Samp_Zj ~ N(Zμj|y, σj), j∼U(1,LatDim)
    - In the expression j∼U(1,LatDim), j corresponds to LatDim and all js are selected randomly.

    '''
    
    # Masking for selecting Samp_Zj from Samp_Z 
    if Axis ==1: #It is strongly recommended when there is no ancillary data inputs or variations in the ancillary data.
        Mask_Z = np.zeros((NMiniBat*NGen, LatDim))
        for i in range(NMiniBat*NGen):
            Mask_Z[i, np.random.choice(LatDim, NSelZ,replace=False )] = 1
            
    elif Axis ==2: # It is strongly recommended in cases where there are variations in ancillary data inputs.
        Mask_Z = np.zeros((NMiniBat, NGen, LatDim))
        for i in range(NMiniBat):
            Mask_Z[i, :, np.random.choice(LatDim, NSelZ,replace=False )] = 1
    
    # Selecting Samp_Zj from Samp_Z 
    Mask_Z = Mask_Z.reshape(NMiniBat*NGen, LatDim)
    Samp_Zj = Samp_Z * Mask_Z
    
    return Samp_Zj




        