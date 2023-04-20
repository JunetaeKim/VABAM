import numpy as np
from scipy.stats import spearmanr

def GenSig_zVar (FeatGenModel, ReconModel, FC, N_Gen=200):
    LatDim= FeatGenModel.input[-1].shape[-1]
    Z_pred = np.linspace(-1.5, 1.5, N_Gen*LatDim).reshape(N_Gen, -1)
    FC_Comm = np.tile(np.ones(2) * FC, (N_Gen,1))
    FC_Each = np.tile(np.ones(4) * FC, (N_Gen,1))
    FeatGen = FeatGenModel([FC_Comm,FC_Each, Z_pred])
    PredFCs = np.concatenate([FC_Comm,FC_Each], axis=-1)
    SigGen = ReconModel([FeatGen])

    HalfLen = SigGen.shape[1]//2
    FFTRes = np.abs(np.fft.fft(SigGen, axis=-1)[:, :HalfLen]) 
    Amplitude = FFTRes/HalfLen
    
    return SigGen, Amplitude


def GenSig_FcVar (FeatGenModel, ReconModel, zValue, N_Gen=200, zType='Fixed'):
    LatDim= FeatGenModel.input[-1].shape[-1]
    if zType=='Random':
        Z_pred=np.random.normal(0, 1, ( N_Gen, LatDim))
    elif zType=='Line' :
        Z_pred = np.linspace(-1.5, 1.5, N_Gen*LatDim).reshape(N_Gen, -1)
    elif zType=='Fixed':
        Z_pred = zValue
        
        
    FC_Comm = np.tile(np.linspace(1e-7, 0.05, N_Gen )[:, None], (1,2))
    FC_Each = np.tile(np.linspace(1e-7, 0.05, N_Gen )[:, None], (1,4))

    FeatGen = FeatGenModel([FC_Comm,FC_Each, Z_pred])
    PredFCs = np.concatenate([FC_Comm,FC_Each], axis=-1)
    SigGen = ReconModel([FeatGen])

    HalfLen = SigGen.shape[1]//2
    FFTRes = np.abs(np.fft.fft(SigGen, axis=-1)[:, :HalfLen]) 
    Amplitude = FFTRes/HalfLen
    
    return SigGen, Amplitude


def GenSig_ZFcVar (FeatGenModel, ReconModel, Z_pred, FC_Comm, FC_Each, FreqRange=[9,12]):

    #[GenSig_ZFcVar(zValues[i], FCValues[i]) for i in range(N_Gen)]
    FeatGen = FeatGenModel([FC_Comm,FC_Each, Z_pred])
    PredFCs = np.concatenate([FC_Comm,FC_Each], axis=-1)
    SigGen = ReconModel([FeatGen])

    HalfLen = SigGen.shape[1]//2
    FFTRes = np.abs(np.fft.fft(SigGen, axis=-1)[:, :HalfLen]) 
    Amplitude = FFTRes/HalfLen
    
    return Amplitude[:, FreqRange[0]:FreqRange[1]+1].max(axis=-1)



def KernelGen (SortedWindow, MinIdx, WindowSize):
    
    TotalSpace = WindowSize-1
    LeftSpaceSize = MinIdx - 0 
    RightSpaceSize = WindowSize-LeftSpaceSize-1
    InterIDX = 1


    LeftSide = []
    RightSide = []

    # Kernel generation
    while LeftSpaceSize>0 or RightSpaceSize>0:

        if LeftSpaceSize>0:
            LeftSide.append(SortedWindow[InterIDX])
            InterIDX += 1
            LeftSpaceSize -= 1

        if RightSpaceSize>0:
            RightSide.append(SortedWindow[InterIDX])
            InterIDX += 1
            RightSpaceSize -= 1

    Kernel = np.concatenate([LeftSide[::-1], [SortedWindow[0]], RightSide])    
    
    return Kernel


def MonotonDegree (Vec, WindowSize):
    
    AbsRhoList = []
    Global_Variability = np.std(Vec)

    for i in range(len(Vec) - WindowSize+1):
        Window = Vec[i:i + WindowSize]

        # Create an ideal monotonic list based on the sorted vector
        SortedWindow = sorted(Window)

        SubRhos=[]
        # Compute Spearman's rho between Kernel and window
        for MinIdx in range(WindowSize):
            LeftSpaceSize = MinIdx - 0 

            Kernel = KernelGen(SortedWindow, MinIdx, WindowSize)
            #Kernel = sorted(Window)

            # Compute Spearman's rho
            SubRho = spearmanr(Window, Kernel)[0]
            # Appeding sub-Rho values
            SubRhos.append(abs(SubRho) * np.log(WindowSize)) # log(WindowSize) : Weighting by increasing window size

        AbsRhoList.append(max(SubRhos))


    return np.mean(AbsRhoList) #* Global_Variability # Global_Variability : Weighting by increasing total std.


## Permutation Local Correlation Monotonicity Index (PLCMI)
def PLCMI (FeatGenModel,  ReconModel, LatDim, N_Gen=300, N_Interval=10, MonoWinSize=20, MinZval = -3., MaxZval = 3., MinFreq=1, MaxFreq=51):
    
    zZeros = np.tile(np.zeros(LatDim), (N_Gen, 1))
    zValues = np.linspace(MinZval , MaxZval , N_Interval)
    
    ResList = []
    
    for zIdx in range(zZeros.shape[1]):
        for zVal in zValues:
            zZeros[:, zIdx] = zVal

            Amplitude_FcVar = GenSig_FcVar(FeatGenModel,  ReconModel, zZeros, N_Gen=N_Gen, zType='Fixed')[1]
            Heatmap = Amplitude_FcVar[:, MinFreq:MaxFreq]
            MaxIDX = np.argmax(np.mean(Heatmap, axis=0))
            Vec =  Heatmap[:, MaxIDX]

            print(zIdx, np.round(zVal,2),  np.round(MonotonDegree(Vec, MonoWinSize), 4))
            
            ResList.append([zIdx, zVal, MonotonDegree(Vec, MonoWinSize)])
            
    return ResList