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
    # (NMiniBat, NGen, SigDim)
    Data = Data[:,None] if len(Data.shape) < 3 else Data

    # Power Spectral Density
    HalfLen = Data.shape[-1]//2
    FFTRes = np.abs(np.fft.fft(Data, axis=-1)[..., :HalfLen])[..., MinFreq:MaxFreq]
    # (NMiniBat, NGen, N_frequency)
    PSD = (FFTRes**2)/Data.shape[-1]

    # Probability Density Function
    if ReducedAxis == 'All':
        AggPSD = np.mean(PSD, axis=(0,1))
        # (N_frequency,)
        AggPSPDF = AggPSD / np.sum(AggPSD, axis=(-1),keepdims=True)
    
    elif ReducedAxis =='Sample':
        AggPSD = np.mean(PSD, axis=(1))
        # (NMiniBat, N_frequency)
        AggPSPDF = AggPSD / np.sum(AggPSD, axis=(-1),keepdims=True)
    
    elif ReducedAxis == 'None':
        # (NMiniBat, NGen, N_frequency)
        AggPSPDF = PSD / np.sum(PSD, axis=(-1),keepdims=True)    
        
    return AggPSPDF


# Permutation given PSD over each generation
def ProbPermutation(Data, WindowSize=3):
    # For the M generation vectors, Data shape: (NMiniBat, N_frequency, NGen)
    # For the true PSD, Data shape: (1, N_frequency, NMiniBat)
    
    # Generating true permutation cases
    TruePerms = np.concatenate(list(itertools.permutations(np.arange(WindowSize)))).reshape(-1, WindowSize)

    # Getting all permutation cases
    Data_Ext = tf.signal.frame(Data, frame_length=WindowSize, frame_step=1, axis=-1)
    PermsTable =  np.argsort(Data_Ext, axis=-1)

    CountPerms = 1- (TruePerms[None,None,None] == PermsTable[:,:,:, None])
    CountPerms = 1-np.sum(CountPerms, axis=-1).astype('bool')
    # Reducing the window axis
    CountPerms = np.sum(CountPerms, axis=(2))
    
    # Data shape: (NMiniBat, N_frequency, N_permutation_cases)
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



def SamplingZ (Data, SampModel, NMiniBat, NGen, BatchSize = 1000, GPU=True, SampZType='GaussBRpt', SecDataType=None, ReparaStdZj=1.):
    
    '''
    Sampling Samp_Z 

    - Shape of UniqSamp_Z: (NMiniBat, LatDim)
    - UniqSamp_Z ~ N(Zμ|y, σ) or N(Zμ|y, cond, σ) for Type =='Model*'
    - UniqSamp_Z ~ N(Zμ|y, cond, σ) for Type =='Model*' and when there are ancillary (i.e., Conditional VAE) data inputs 
    - RandSamp_Z ~ N(0, ReparaStdZj) for Type =='Gauss*'

    - Samp_Z is a 3D tensor expanded by repeating the first axis (i.e., 0) of UniqSamp_Z or RandSamp_Z by NGen times.
    - Shape of Samp_Z: (NMiniBat, NGen, LatDim) -> (NMiniBat*NGen, LatDim) for optimal use of GPU 
    
    - ModelBRpt:  The predicted values are repeated NGen times after the prediction. 
                  It is strongly recommended in cases where there are variations in the ancillary data inputs.  
    - ModelARand: The data is repeated NGen times before the prediction (i.e., All random sampling).
                  It is strongly recommended when there is no ancillary data inputs or variations in the ancillary data.     
    - Gauss:      The data is sampled NMiniBat*NGen times. 
                  It is recommended to detect the influence of changes in the value of j on the performance metrics, 
                  when the same ancillary data input is repeated NGen times (i.e., for Conditional VAE).
    - GaussBRpt:  The data sampled from the Gaussian distribution is repeated NGen times.
                  It is recommended for both cases: when there are no ancillary data inputs and when there is ancillary data input.
                  
    '''
    assert SampZType in ['ModelBRpt','ModelARand','Gauss', 'GaussBRpt'], "Please verify the value of 'SampZType'. Only 'ModelBRpt','ModelARand','Gauss', 'GaussBRpt' are valid."
    
    # Sampling Samp_Z
    if SampZType =='ModelBRpt': # Z ~ N(Zμ|y, σ) or N(Zμ|y, cond, σ) 
        UniqSamp_Z = Sampler(Data, SampModel, GPU=GPU)
        Samp_Z =  np.broadcast_to(UniqSamp_Z[:, None], (NMiniBat, NGen, UniqSamp_Z.shape[-1])).reshape(-1, UniqSamp_Z.shape[-1])
    
    elif SampZType =='ModelARand': # Z ~ N(Zμ|y, σ) or N(Zμ|y, cond, σ)
        
        if SecDataType == 'CONA' or SecDataType == 'CONR' : # For the CondVAE
            DataRpt = [np.repeat(arr, NGen, axis=0) for arr in Data]
        else:
            DataRpt = np.repeat(Data, NGen, axis=0)
        Samp_Z = Sampler(DataRpt, SampModel, GPU=GPU)
        
    elif SampZType =='Gauss': # Z ~ N(0, ReparaStdZj)
        Samp_Z = np.random.normal(0, ReparaStdZj, (NMiniBat*NGen , SampModel.output.shape[-1]))
        
    elif SampZType =='GaussBRpt': # Z ~ N(0, ReparaStdZj)
        UniqSamp_Z = np.random.normal(0, ReparaStdZj, (NMiniBat , SampModel.output.shape[-1]))
        Samp_Z = np.repeat(UniqSamp_Z, NGen, axis=0)
    
    return Samp_Z



def SamplingZj (Samp_Z, NMiniBat, NGen, LatDim, NSelZ, ZjType='ARand' ):
    
    '''
     Sampling Samp_Zj 

    - Masking is applied to select Samp_Zj from Samp_Z 
      by assuming that the Samp_Z with indices other than j have a fixed mean value of '0' following a Gaussian distribution.

    - Shape of Samp_Zj: (NMiniBat, NGen, LatDim) -> (NMiniBat*NGen, LatDim) for optimal use of GPU 

    - Samp_Zj ~ N(Zμj|y, σj), j∼U(1,LatDim)
    - In the expression j∼U(1,LatDim), j corresponds to LatDim and all js are selected randomly.

    '''
    
    # Masking for selecting Samp_Zj from Samp_Z 
    if ZjType =='ARand': #It is strongly recommended when there is no ancillary data inputs or variations in the ancillary data.
        Mask_Z = np.zeros((NMiniBat*NGen, LatDim))
        for i in range(NMiniBat*NGen):
            Mask_Z[i, np.random.choice(LatDim, NSelZ,replace=False )] = 1
            
    elif ZjType =='BRpt': # It is strongly recommended in cases where there are variations in ancillary data inputs.
        Mask_Z = np.zeros((NMiniBat, NGen, LatDim))
        for i in range(NMiniBat):
            Mask_Z[i, :, np.random.choice(LatDim, NSelZ,replace=False )] = 1
    
    # Selecting Samp_Zj from Samp_Z 
    Mask_Z = Mask_Z.reshape(NMiniBat*NGen, LatDim)
    Samp_Zj = Samp_Z * Mask_Z
    
    return Samp_Zj



def SamplingFCs (NMiniBat, NGen, NFCs, FCexist=None, SampFCType='ARand', FcLimit= 0.05):

    # Check for valid SampFCType values
    if SampFCType not in ['ARand', 'BRpt', 'Sort']:
        raise ValueError(f"Invalid SampFCType: {SampFCType}. Expected one of: 'ARand', 'BRpt', 'Sort'")
    
    # Sampling FCs
    ## Return shape of FCs: (NMiniBat*NGen, NFCs) instead of (NMiniBat, NGen, NFCs) for optimal use of GPU
    if SampFCType =='ARand':
        FCs = np.random.rand(NMiniBat*NGen, NFCs) * FcLimit
    elif SampFCType == 'BRpt' :
        FCs = np.random.rand(NMiniBat,  NFCs) * FcLimit
        FCs = np.repeat(FCs, NGen, axis=0)

    # For Sampling FC_organized 
    ## Return shape of FCs:(NMiniBat, NGen, NFCs) instead of (NMiniBat*NGen, NFCs)
    if FCexist is None:
        FCexist = FCs
        FCexist = FCexist.reshape(NMiniBat, NGen, NFCs)

    
    # Sampling FC_organized (i.e., FCor)
    if SampFCType == 'Sort' :
        FCs =np.sort(FCexist, axis=0).reshape(NMiniBat*NGen, NFCs)

    return FCs
    

def GenConArange (ConData, NGen):
    # Processing Conditional information 
    ## Finding the column index of the max value in each row of ConData and sort the indices
    ArgMaxP_PSPDF = np.argmax(ConData, axis=-1)
    SortIDX = np.column_stack((np.argsort(ArgMaxP_PSPDF), ArgMaxP_PSPDF[np.argsort(ArgMaxP_PSPDF)]))

    # Computing the number of iterations
    UniqPSPDF = np.unique(ArgMaxP_PSPDF)
    NIter = NGen // len(UniqPSPDF)

    # Selecting one row index for each unique value, repeated for NIter times and ensuring the total number of selected indices matches NGen
    SelIDX = np.concatenate([np.random.permutation(SortIDX[SortIDX[:, 1] == psd])[:1] for psd in UniqPSPDF for _ in range(NIter)], axis=0)
    SelIDX = np.vstack((SelIDX, np.random.permutation(SortIDX)[:NGen - len(SelIDX)]))

    # Sorting IDX based on the max values
    SelIDX = SelIDX[np.argsort(SelIDX[:, 1])]

    ## Generating CON_Arange
    return ConData[SelIDX[:,0]]




        
