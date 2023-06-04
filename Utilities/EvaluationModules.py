import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import EntropyHub as EH


def GenSig_zVar (FeatGenModel, ReconModel, FC, N_Gen=200, MinZval = -3., MaxZval = 3.,):
    LatDim= FeatGenModel.input[-1].shape[-1]
    Z_pred = np.linspace(MinZval, MaxZval, N_Gen*LatDim).reshape(N_Gen, -1)
    FC_Comm = np.tile(np.ones(2) * FC, (N_Gen,1))
    FC_Each = np.tile(np.ones(4) * FC, (N_Gen,1))
    FeatGen = FeatGenModel([FC_Comm,FC_Each, Z_pred])
    PredFCs = np.concatenate([FC_Comm,FC_Each], axis=-1)
    SigGen = ReconModel([FeatGen])

    HalfLen = SigGen.shape[1]//2
    FFTRes = np.abs(np.fft.fft(SigGen, axis=-1)[:, :HalfLen]) 
    Amplitude = FFTRes/HalfLen
    
    return SigGen, Amplitude


def GenSig_FcVar (FeatGenModel, ReconModel, zValue, N_Gen=200, MaxFreq =0.05, MinZval = -3., MaxZval = 3., zType='Fixed'):
    LatDim= FeatGenModel.input[-1].shape[-1]
    if zType=='Random':
        Z_pred=np.random.normal(0, 1, ( N_Gen, LatDim))
    elif zType=='Line' :
        Z_pred = np.linspace(MinZval, MaxZval, N_Gen*LatDim).reshape(N_Gen, -1)
    elif zType=='Fixed':
        Z_pred = zValue
        
        
    FC_Comm = np.tile(np.linspace(1e-7, MaxFreq, N_Gen )[:, None], (1,2))
    FC_Each = np.tile(np.linspace(1e-7, MaxFreq, N_Gen )[:, None], (1,4))

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



def MonotonDegree (Vec, WindowSize, Weight, Type):
    
    MDList = []
    Global_Variability = np.std(Vec)
    
    for i in range(len(Vec) - WindowSize+1):
        
        Window = Vec[i:i + WindowSize]
        # Create an ideal monotonic list based on the sorted vector
        SortedWindow = sorted(Window).copy()

        SubMDs=[]
        # Compute MD values between Kernel and window
        for MinIdx in range(WindowSize):
            LeftSpaceSize = MinIdx - 0 

            Kernel = KernelGen(SortedWindow, MinIdx, WindowSize)
            #Kernel = sorted(Window)
            
            if Type == 'Cor':
                # Compute Spearman's rho
                SubMD = spearmanr(Window, Kernel)[0]
            elif Type == 'L2':               
                SubMD = np.sum((Window- Kernel)**2)
                SubMD = np.log(1/SubMD)
            elif Type == 'R2':       
                SubMD = np.maximum(r2_score(Window, Kernel), 0) 
                
            # Appeding sub-MD values
            if Weight == 'WindowSize':
                WeightVal = WindowSize
            elif Weight == 'Continuity':
                WeightVal = np.maximum(WindowSize - np.argmin(Kernel), np.argmin(Kernel)+1)
            elif Weight == 'Amplitude':
                WeightVal = np.log(np.sum(Kernel))
            elif Weight == 'RevHHI':
                KernelRate = Kernel / np.sum(Kernel)
                WeightVal = np.exp(1-np.sum(KernelRate**2))
            elif Weight == None:
                WeightVal = 1.
                

            SubMDs.append(abs(SubMD) * WeightVal) # np.log(Continuity) or log(WindowSize) : Weighting by increasing window size

        MDList.append(max(SubMDs))


    return np.mean(MDList) #* Global_Variability # Global_Variability : Weighting by increasing total std.


## Permutation Local Monotonicity Index (PLMI)
def PLMI (FeatGenModel,  ReconModel, LatDim, N_Gen=300, N_Interval=10, MonoWinSize=20, MinZval = -3., MaxZval = 3., N_FreqSel =3 , MinFreq=1, MaxFreq=51, Weight=None, Type='Cor'):
    assert Weight in ['WindowSize','Continuity', 'Amplitude', 'RevHHI', None], '''either 'WindowSize', 'Continuity','Amplitude' or 'RevHHI' is allowed for 'weight' '''
    
    zValues = np.linspace(MinZval , MaxZval , N_Interval)
    
    ResList = []
    for zIdx in range(LatDim):
        zZeros = np.tile(np.zeros(LatDim), (N_Gen, 1))
        for zVal in zValues:
            zZeros[:, zIdx] = zVal

            Amplitude_FcVar = GenSig_FcVar(FeatGenModel,  ReconModel, zZeros, N_Gen=N_Gen, zType='Fixed')[1]
            Heatmap = Amplitude_FcVar[:, MinFreq:MaxFreq]
            MaxIDX = np.argsort(np.mean(Heatmap, axis=0))[-N_FreqSel:]
            
            MonoRes = []
            for IDX in MaxIDX:
                Vec =  Heatmap[:, IDX]
                MonoRes.append(np.round(MonotonDegree(Vec, MonoWinSize, Weight, Type), 4)) 
            
            Res = [zIdx, np.round(zVal,2), np.max(MonoRes)]
                
            print(Res)
            
            ResList.append(Res)
            
    return ResList





### Qualitative and Visual Evaluation
def HeatMapFrequency (FeatGenModel,  ReconModel, LatDim, ZFix, N_Gen=300, MinFreq=1, MaxFreq=51):
    
    zVal = np.tile(np.zeros(LatDim), (N_Gen,1))
    for KeyVal in ZFix.items():
        zVal[:,KeyVal[0]] = KeyVal[1]
    
    SigGen_FcVar, Amplitude_FcVar = GenSig_FcVar(FeatGenModel,  ReconModel, zVal, N_Gen=N_Gen, zType='Fixed')
    Heatmap = Amplitude_FcVar[:, MinFreq:MaxFreq]

    fig, ax = plt.subplots(figsize=(7,6))
    cax = fig.add_axes([0.95, 0.25, 0.04, 0.5])

    im = ax.imshow(Heatmap,  cmap='viridis', aspect='auto',interpolation='nearest') 
    ax.set(yticks=np.arange(1, N_Gen)[::10], yticklabels=np.round(np.linspace(1e-7, 0.05, N_Gen )[::10]*100, 1));
    ax.set(xticks=np.arange(1, MaxFreq)[::5]-0.5, xticklabels=np.arange(1, MaxFreq)[::5]);
    ax.set_xlabel('Frequency', fontsize=16)
    ax.set_ylabel('Frequency given for generating signals', fontsize=16) 

    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.show()
    
    
def VisReconExtract (ValData, idx, FeatExtModel, ReconModel, FC_Comm, FC_Each, N_Gen=300):
    
    Sample = np.tile(ValData[idx][None], (N_Gen, 1))
    print(Sample.shape)
    FCs = np.concatenate([FC_Comm, FC_Each], axis=-1)
    HH,HL,LH, LL = FeatExtModel([Sample, FCs])
    RecPred = ReconModel([HH,HL,LH, LL])

    plt.figure(figsize=(15, 5))
    for i in range(N_Gen):
        plt.plot(RecPred[i])
    plt.plot(ValData[idx],linestyle='--', color='black', label='True signal')
    plt.legend()
    
    return RecPred, HH,HL,LH, LL


def VisReconGivenZ (FeatGenModel,  ReconModel, LatDim, ZFix, Mode='Origin', N_Gen=300, MinFreqR=0., MaxFreqR=0.05):
    
    assert Mode in ['Origin','HH','HL','LH','LL'], '''either 'Origin', 'HH', 'HL', 'LH', and 'LL' is allowed for 'Mode' '''

    zVal = np.tile(np.zeros(LatDim), (N_Gen,1))
    for KeyVal in ZFix.items():
        zVal[:,KeyVal[0]] = KeyVal[1]
        
    FC_Comm = np.tile(np.linspace(MinFreqR, MaxFreqR, N_Gen )[:, None], (1,2))
    FC_Each = np.tile(np.linspace(MinFreqR, MaxFreqR, N_Gen )[:, None], (1,4))

    FeatGen = FeatGenModel([FC_Comm,FC_Each, zVal])
    PredFCs = np.concatenate([FC_Comm,FC_Each], axis=-1)
    SigGen = ReconModel([FeatGen])
    
    
    # Create a colormap and normalize it based on the number of experiments
    cmap = cm.get_cmap('viridis')
    norm = plt.Normalize(0, N_Gen-1)
    norm2 = plt.Normalize(MinFreqR, MaxFreqR)

    if Mode == 'Origin':
        VisSet = SigGen
    elif Mode == 'HH':
        VisSet = FeatGen[0]
    elif Mode == 'HL':
        VisSet = FeatGen[1]
    elif Mode == 'LH':
        VisSet = FeatGen[2]
    elif Mode == 'LL':
        VisSet = FeatGen[3]    

    fig, ax = plt.subplots(figsize=(15, 7))
    for i in range(0, N_Gen):
        color = cmap(norm(i))
        ax.plot(VisSet[i], color=color)


    # Create color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm2)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Frequency ranges', size=14)

    plt.show()
    
    return SigGen, FeatGen[0], FeatGen[1], FeatGen[2], FeatGen[3]

    
    
def VisReconGivenFreq (FeatGenModel,  ReconModel, LatDim, FcCommFix, FcEachFix,  Mode='Origin', N_Gen=300, MinZval = -3., MaxZval = 3., CutLower=-0.1, CutUpper = 0.1):
    
    zVal = np.linspace(MinZval, MaxZval, N_Gen*LatDim).reshape(N_Gen, -1)

    FC_Comm = np.tile(np.zeros(2) , (N_Gen,1))
    FC_Each = np.tile(np.zeros(4) , (N_Gen,1))

    for KeyVal in FcCommFix.items():
        FC_Comm[:,KeyVal[0]] = KeyVal[1]

    for KeyVal in FcEachFix.items():
        FC_Each[:,KeyVal[0]] = KeyVal[1]
        
        
    FeatGen = FeatGenModel([FC_Comm,FC_Each, zVal])
    PredFCs = np.concatenate([FC_Comm,FC_Each], axis=-1)
    SigGen = ReconModel([FeatGen])
    
    # Create a colormap and normalize it based on the number of experiments
    cmap = cm.get_cmap('viridis')
    norm = plt.Normalize(0, N_Gen-1)
    norm2 = plt.Normalize(MinZval, MaxZval)

    if Mode == 'Origin':
        VisSet = SigGen
    elif Mode == 'HH':
        VisSet = FeatGen[0]
    elif Mode == 'HL':
        VisSet = FeatGen[1]
    elif Mode == 'LH':
        VisSet = FeatGen[2]
    elif Mode == 'LL':
        VisSet = FeatGen[3]    

    fig, ax = plt.subplots(figsize=(15, 7))
    for i in range(0, N_Gen):
        color = cmap(norm(i))
        if np.min(zVal[i]) < CutLower or np.max(zVal[i]) > CutUpper:
            ax.plot(VisSet[i], color=color)


    # Create color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm2)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('z-value ranges')

    plt.show()
    
    return SigGen, FeatGen[0], FeatGen[1], FeatGen[2], FeatGen[3]



## Local Permutation Entropy Index (LPEI)
def LPEI (FeatGenModel,  ReconModel, LatDim, N_Gen=300, N_Interval=10, MonoWinSize=20, MinZval = -3., MaxZval = 3., N_FreqSel =3 , MinFreq=1, MaxFreq=51, Weight=None):
    assert Weight in ['WindowSize', 'Amplitude', 'RevHHI', None], '''either 'WindowSize', 'Amplitude' or 'RevHHI' is allowed for 'weight' '''
    
    
    zValues = np.linspace(MinZval , MaxZval , N_Interval)
    
    ResList = []
    for zIdx in range(LatDim):
        zZeros = np.tile(np.zeros(LatDim), (N_Gen, 1))
        for zVal in zValues:
            zZeros[:, zIdx] = zVal

            Amplitude_FcVar = GenSig_FcVar(FeatGenModel,  ReconModel, zZeros, N_Gen=N_Gen, zType='Fixed')[1]
            Heatmap = Amplitude_FcVar[:, MinFreq:MaxFreq]
            MaxIDX = np.argsort(np.mean(Heatmap, axis=0))[-N_FreqSel:]
            
            MonoRes = []
            for IDX in MaxIDX:
                             
                Kernel = Heatmap[:, IDX]
                
                # Appeding sub-MD values
                if Weight == 'WindowSize':
                    WeightVal = MonoWinSize
                elif Weight == 'Amplitude':
                    WeightVal = np.log(np.sum(Kernel))
                elif Weight == 'RevHHI':
                    KernelRate = Kernel / np.sum(Kernel)
                    WeightVal = np.exp(1-np.sum(KernelRate**2))
                elif Weight == None:
                    WeightVal = 1.

                MonoRes.append( (1/EH.PermEn(Kernel)[0][1]) * WeightVal ) 
                
            Res = [zIdx, np.round(zVal,2), np.min(MonoRes), MaxIDX.tolist()]
                
            print(Res)
            
            ResList.append(Res)
            
    return ResList