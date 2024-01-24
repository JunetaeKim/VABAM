import numpy as np
from scipy.stats import mode
import itertools
from tqdm import trange, tqdm
import gc
import tensorflow as tf
from tensorflow.keras import Model

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Utilities.Utilities import GenBatches


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
    Data = Data[:,None] if len(Data.shape) < 3 else Data

    # Power Spectral Density
    HalfLen = Data.shape[-1]//2
    FFTRes = np.abs(np.fft.fft(Data, axis=-1)[..., :HalfLen])[..., MinFreq:MaxFreq]
    # (N, M, N_frequency)
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



# The 'predict' function in TensorFlow version 2.10 may cause memory leak issues.
def Sampler (Data, SampModel,BatchSize=100, GPU=True):
    if GPU==False:
        with tf.device('/CPU:0'):
            PredVal = SampModel.predict(Data, batch_size=BatchSize, verbose=1)   
    else:
        PredVal = SampModel.predict(Data, batch_size=BatchSize, verbose=1)   

    return PredVal

'''
def Sampler (Data, SampModel,BatchSize=100, GPU=True):
    Data = [Data]
    PredVal = []
    TotalBatches = len(Data[0]) // BatchSize + (len(Data[0]) % BatchSize != 0)

    if GPU==False:
        with tf.device('/CPU:0'):
            for i, Batch in enumerate(GenBatches(Data, BatchSize)):
                BatchPred = SampModel.predict_on_batch(Batch[0])
                print(f"Processing batch {i+1}/{TotalBatches}", end='\r')
                PredVal.extend(BatchPred)
                gc.collect()
    else:
        for i, Batch in enumerate(GenBatches(Data, BatchSize)):
            BatchPred = SampModel.predict_on_batch(Batch[0])
            print(f"Processing batch {i+1}/{TotalBatches}", end='\r')
            PredVal.extend(BatchPred)
            gc.collect()
            
    return np.array(PredVal)
'''

def SamplingZ (Data, SampModel, NMiniBat, NParts, NSubGen, BatchSize = 1000, GPU=True, SampZType='Modelbd',  SecDataType=None, ReparaStdZj=1.):
    
    '''
    Sampling Samp_Z 
    - Return shape of Samp_Z: (NMiniBat, NGen, LatDim) -> (NMiniBat*NGen, LatDim) for optimal use of GPU 
    - NGen = NParts * NSubGen
    
    - Modelbd: The predicted values are repeated NGen times after the prediction. 
    - Modelbr: The data is repeated NParts times before the prediction. Then, The predicted values are repeated NSubGen times .
    - Gaussbr: The data sampled (NMiniBat, NParts, LatDim) from the Gaussian distribution is repeated NSubGen times. 
    '''
    
    assert SampZType in ['Modelbd','Modelbr','Gaussbr'], "Please verify the value of 'SampZType'. Only 'Modelbd','Modelbr','Gaussbr' are valid."
    NGen = NParts * NSubGen
    
    # Sampling Samp_Z
    if SampZType =='Modelbd': # Z ~ N(Zμ|y, σ) or N(Zμ|y, cond, σ) 
        # Shape of UniqSamp_Z: (NMiniBat, LatDim) 
        UniqSamp_Z = Sampler(Data, SampModel, BatchSize=BatchSize, GPU=GPU)
        Samp_Z =  np.broadcast_to(UniqSamp_Z[:, None], (NMiniBat, NGen, UniqSamp_Z.shape[-1])).reshape(-1, UniqSamp_Z.shape[-1])
        
    elif SampZType =='Modelbr':
        DataExt = np.repeat(Data, NParts, axis=0)
        # Shape of UniqSamp_Z: (NMiniBat, NParts, LatDim) 
        UniqSamp_Z = Sampler(DataExt, SampModel, BatchSize=BatchSize, GPU=GPU).reshape(NMiniBat, NParts, -1)
        Samp_Z =  np.broadcast_to(UniqSamp_Z[:, :, None], (NMiniBat, NParts, NSubGen, UniqSamp_Z.shape[-1])).reshape(-1, UniqSamp_Z.shape[-1])

    elif SampZType =='Gaussbr': # Z ~ N(0, ReparaStdZj)
        # Shape of UniqSamp_Z: (NMiniBat, NParts, LatDim) 
        UniqSamp_Z = np.random.normal(0, ReparaStdZj, (NMiniBat, NParts , SampModel.output.shape[-1]))
        Samp_Z =  np.broadcast_to(UniqSamp_Z[:, :, None], (NMiniBat, NParts, NSubGen, UniqSamp_Z.shape[-1])).reshape(-1, UniqSamp_Z.shape[-1])

    # Return shape of Samp_Z: (NMiniBat*NGen, LatDim)
    return Samp_Z


def SamplingZj (Samp_Z, NMiniBat, NGen, LatDim, NSelZ, ZjType='bd' ):
    
    '''
     Sampling Samp_Zj 
    - Return shape of Samp_Zj: (NMiniBat, NGen, LatDim) -> (NMiniBat*NGen, LatDim) for optimal use of GPU 
    - Masking is applied to select Samp_Zj from Samp_Z 
      by assuming that the Samp_Z with indices other than j have a fixed mean value of '0' following a Gaussian distribution.
    - Samp_Zj ~ N(Zμj|y, σj), j∼U(1,LatDim)
    - In the expression j∼U(1,LatDim), j corresponds to LatDim and all js are selected randomly.

    '''
    
    # Masking for selecting Samp_Zj from Samp_Z 
    if ZjType =='bd': 
        Mask_Z = np.zeros((NMiniBat, NGen, LatDim))
        for i in range(NMiniBat):
            Mask_Z[i, :, np.random.choice(LatDim, NSelZ,replace=False )] = 1
    
    # Selecting Samp_Zj from Samp_Z 
    Mask_Z = Mask_Z.reshape(NMiniBat*NGen, LatDim)
    Samp_Zj = Samp_Z * Mask_Z

    # Return shape of Samp_Zj: (NMiniBat*NGen, LatDim)
    return Samp_Zj



def SamplingFCs (NMiniBat, NGen, NFCs, SampFCType='ARand', FcLimit= 0.05):

    # Check for valid SampFCType values
    if SampFCType not in ['ARand', 'BRpt']:
        raise ValueError(f"Invalid SampFCType: {SampFCType}. Expected one of: 'ARand', 'BRpt' ")
    
    # Sampling FCs
    ## Return shape of FCs: (NMiniBat*NGen, NFCs) instead of (NMiniBat, NGen, NFCs) for optimal use of GPU
    if SampFCType =='ARand':
        FCs = np.random.rand(NMiniBat*NGen, NFCs) * FcLimit
    elif SampFCType == 'BRpt' :
        FCs = np.random.rand(NMiniBat,  NFCs) * FcLimit
        FCs = np.repeat(FCs, NGen, axis=0)

    return FCs
    

def Partition3D(Mat, NParts):
    B, M, F = Mat.shape
    PartSize = M // NParts
    Remainder = M % NParts

    NewMat = np.zeros_like(Mat)
    # ReturnIDX will store the partition ID and local position for each element
    ReturnIDX = np.zeros((B, M, 2), dtype=int)  # Adding an extra dimension for partition ID and local position

    for b in range(B):
        CumulativeIndex = 0
        for i in range(NParts):
            PartSizeAdjusted = PartSize + (1 if i < Remainder else 0)
            Slice = Mat[b, CumulativeIndex:CumulativeIndex + PartSizeAdjusted, :]
            SortedIndices = np.argsort(Slice, axis=0)
            SortedSlice = np.take_along_axis(Slice, SortedIndices, axis=0)

            NewMat[b, CumulativeIndex:CumulativeIndex + PartSizeAdjusted, :] = SortedSlice
            # Store partition ID and local position
            ReturnIDX[b, CumulativeIndex:CumulativeIndex + PartSizeAdjusted, 0] = i
            for f in range(F):
                ReturnIDX[b, CumulativeIndex:CumulativeIndex + PartSizeAdjusted, 1] = np.arange(PartSizeAdjusted)

            CumulativeIndex += PartSizeAdjusted

    return NewMat, ReturnIDX[:,:,0]


def GenConArange (ConData, NGen):
    # Processing Conditional information 
    ## Finding the column index of the max value in each row of ConData and sort the indices
    ArgMaxP_PSPDF = np.argmax(ConData, axis=-1)
    SortIDX = np.column_stack((np.argsort(ArgMaxP_PSPDF), ArgMaxP_PSPDF[np.argsort(ArgMaxP_PSPDF)]))

    # Computing the number of iterations
    UniqPSPDF = np.unique(ArgMaxP_PSPDF)
    NIter = NGen // len(UniqPSPDF)

    # Selecting one row index for each unique value, repeated for NIter times and ensure the total number of selected indices matches NGen
    SelIDX = np.concatenate([np.random.permutation(SortIDX[SortIDX[:, 1] == psd])[:1] for psd in UniqPSPDF for _ in range(NIter)], axis=0)
    SelIDX = np.vstack((SelIDX, np.random.permutation(SortIDX)[:NGen - len(SelIDX)]))

    # Sorting IDX based on the max values
    SelIDX = SelIDX[np.argsort(SelIDX[:, 1])]

    ## Generating CON_Arange
    return ConData[SelIDX[:,0]]


def GenConArangeSimple (ConData, NGen, seed=1):

    ### Selecting Conditional dataset randomly
    np.random.seed(seed)
    ConData = np.random.permutation(ConData)[:NGen]
    
    ### Identifying the index of maximum frequency for each selected condition
    MaxFreqSelCond = np.argmax(ConData, axis=-1)
    ### Sorting the selected conditions by their maximum frequency
    Idx_MaxFreqSelCond = np.argsort(MaxFreqSelCond)
    CONbm_Sort = ConData[Idx_MaxFreqSelCond]

    return CONbm_Sort



def Denorm (NormX, MaxX, MinX):
    return NormX * (MaxX - MinX) + MinX 


def MAPECal (TrueData, PredSigRec, MaxX, MinX):
    # Denormalization
    DenormTrueData = Denorm(TrueData, MaxX, MinX).copy()
    DenormPredSigRec = Denorm(PredSigRec, MaxX, MinX).copy()
   
    # MAPE
    MAPEdenorm = np.mean(np.abs((DenormTrueData - DenormPredSigRec) / DenormTrueData)) * 100
    MAPEnorm = np.mean(np.abs(((TrueData+1e-7) - PredSigRec) / (TrueData+1e-7))) * 100

    return MAPEnorm, MAPEdenorm


def MSECal (TrueData, PredSigRec, MaxX, MinX):
    # Denormalization
    DenormTrueData = Denorm(TrueData, MaxX, MinX).copy()
    DenormPredSigRec = Denorm(PredSigRec, MaxX, MinX).copy()
   
    # MAPE
    MSEdenorm = np.mean((DenormTrueData - DenormPredSigRec)**2)
    MSEnorm = np.mean((TrueData - PredSigRec)**2)
    
    return MSEnorm, MSEdenorm





        
