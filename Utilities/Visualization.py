import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Utilities.AncillaryFunctions import GenConArangeSimple

import tensorflow as tf
from tensorflow.keras import Model


def VisReconGivenZ (ReconModel, zValue,  N_Gen=300):

    SigGen = ReconModel.predict(zValue)
    
    # Create a colormap and normalize it based on the number of experiments
    cmap = cm.get_cmap('viridis')
    norm = plt.Normalize(0, N_Gen-1)
    MeanZs = np.mean(zValue, axis=1)
    norm2 = plt.Normalize(min(MeanZs), max(MeanZs))

    fig, ax = plt.subplots(figsize=(15, 7))
    for i in range(0, N_Gen):
        color = cmap(norm(i))
        ax.plot(SigGen[i], color=color)


    # Create color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm2)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Mean of z for signal generation', size=14)

    plt.show()
    
    return SigGen


### Qualitative and Visual Evaluation
def HeatMapFreqZ ( SigGen, MinFreq=1, MaxFreq=51):
    
    N_Gen = SigGen.shape[0]
    
    HalfLen = SigGen.shape[1]//2
    FFTRes = np.abs(np.fft.fft(SigGen, axis=-1)[:, :HalfLen]) 
    Amplitude = FFTRes/HalfLen
    Heatmap = Amplitude[:, MinFreq:MaxFreq]

    fig, ax = plt.subplots(figsize=(7,6))
    cax = fig.add_axes([0.95, 0.25, 0.04, 0.5])

    im = ax.imshow(Heatmap,  cmap='viridis', aspect='auto',interpolation='nearest') 
    #ax.set(yticks=np.arange(1, N_Gen)[::25], yticklabels=np.round(np.arange(1, N_Gen)[::25], 1));
    ax.set(xticks=np.arange(1, MaxFreq)[::5]-0.5, xticklabels=np.arange(1, MaxFreq)[::5]);
    ax.set_xlabel('Frequency', fontsize=16)
    ax.set_ylabel('Index of signals generated with a series of mean of z', fontsize=13) 

    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.show()

    
    
### ---------------------------------------- For the models with FCs ----------------------------------------  
    
def GenSig_FCA (FeatGenModel, ReconModel, zValue, N_Gen=200, MaxFreqR =0.05, MinZval = -3., MaxZval = 3., zType='Fixed'):
    LatDim= FeatGenModel.input[-1].shape[-1]
    if zType=='Random':
        Z_pred=np.random.normal(0, 1, ( N_Gen, LatDim))
    elif zType=='Line' :
        Z_pred = np.linspace(MinZval, MaxZval, N_Gen*LatDim).reshape(N_Gen, -1)
    elif zType=='Fixed':
        Z_pred = zValue
        
        
    FC_Comm = np.tile(np.linspace(1e-7, MaxFreqR, N_Gen )[:, None], (1,2))
    FC_Each = np.tile(np.linspace(1e-7, MaxFreqR, N_Gen )[:, None], (1,4))

    FeatGen = FeatGenModel([FC_Comm,FC_Each, Z_pred])
    PredFCs = np.concatenate([FC_Comm,FC_Each], axis=-1)
    SigGen = ReconModel([FeatGen])

    HalfLen = SigGen.shape[1]//2
    FFTRes = np.abs(np.fft.fft(SigGen, axis=-1)[:, :HalfLen]) 
    Amplitude = FFTRes/HalfLen
    
    return SigGen, Amplitude

    
def HeatMapFreqZ_FCA (FeatGenModel,  ReconModel, LatDim, ZFix, N_Gen=300, MaxFreqR =0.05, MinFreq=1, MaxFreq=51):
    
    zVal = np.tile(ZFix, (N_Gen,1))
    
    SigGen_FcVar, Amplitude_FcVar = GenSig_FCA(FeatGenModel,  ReconModel, zVal, N_Gen=N_Gen, MaxFreqR=MaxFreqR, zType='Fixed')
    Heatmap = Amplitude_FcVar[:, MinFreq:MaxFreq]

    fig, ax = plt.subplots(figsize=(7,6))
    cax = fig.add_axes([0.95, 0.25, 0.04, 0.5])

    im = ax.imshow(Heatmap,  cmap='viridis', aspect='auto',interpolation='nearest') 
    ax.set(yticks=np.arange(0, N_Gen)[::20]);
    ax.set(xticks=np.arange(1, MaxFreq)[::5]-0.5, xticklabels=np.arange(1, MaxFreq)[::5]);
    ax.set_xlabel('Frequency', fontsize=16)
    ax.set_ylabel('Index of signals generated with a series of $\\acute{\\theta}$', fontsize=13) 

    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.show()
    
    
def VisReconExtractZ_FC (Data, idx, FeatExtModel, ReconModel, FC_Comm, FC_Each, N_Gen=300):
    
    Sample = np.tile(Data[idx][None], (N_Gen, 1))
    print(Sample.shape)
    FCs = np.concatenate([FC_Comm, FC_Each], axis=-1)
    HH,HL,LH, LL = FeatExtModel([Sample, FCs])
    RecPred = ReconModel([HH,HL,LH, LL])

    plt.figure(figsize=(15, 5))
    for i in range(N_Gen):
        plt.plot(RecPred[i])
    plt.plot(Data[idx],linestyle='--', color='black', label='True signal')
    plt.legend()
    
    return RecPred, HH,HL,LH, LL


def VisReconGivenZ_FCA (FeatGenModel,  ReconModel, LatDim, ZFix, Mode='Origin', N_Gen=300, MinFreqR=0., MaxFreqR=0.05):
    
    assert Mode in ['Origin','HH','HL','LH','LL'], '''either 'Origin', 'HH', 'HL', 'LH', and 'LL' is allowed for 'Mode' '''

    zVal = np.tile(ZFix, (N_Gen,1))
        
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
    cbar.set_label('$\\acute{\\theta}$ given for signal generation', size=14)

    plt.show()
    
    return SigGen, FeatGen[0], FeatGen[1], FeatGen[2], FeatGen[3]

    
def VisReconGivenFC_ZA (FeatGenModel,  ReconModel, LatDim, FcCommFix, FcEachFix,  Mode='Origin', N_Gen=300, MinZval = -3., MaxZval = 3., CutLower=-0.1, CutUpper = 0.1):
    
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




### ---------------------------------------- For ConVAE ----------------------------------------

def GenSig_CONA (ReconModel, zValue, ConData, N_Gen=200, MinZval = -3., MaxZval = 3., zType='Fixed'):

    if zType=='Random':
        Z_pred=np.random.normal(0, 1, ( N_Gen, LatDim))
    elif zType=='Line' :
        Z_pred = np.linspace(MinZval, MaxZval, N_Gen*LatDim).reshape(N_Gen, -1)
    elif zType=='Fixed':
        Z_pred = zValue
        
        
    ## Generating CON_Arange
    CON_Arange = GenConArangeSimple(ConData, N_Gen)

    SigGen = ReconModel([Z_pred, CON_Arange])

    HalfLen = SigGen.shape[1]//2
    FFTRes = np.abs(np.fft.fft(SigGen, axis=-1)[:, :HalfLen]) 
    Amplitude = FFTRes/HalfLen
    
    return SigGen, Amplitude


def HeatMapFreqZ_CONA (ReconModel, ConData, LatDim, ZFix, N_Gen=300, MinFreq=1, MaxFreq=51):
    
    zVal = np.tile(ZFix, (N_Gen,1))
    
    _, Amplitude_ConVar = GenSig_CONA(ReconModel, zVal, ConData, N_Gen=N_Gen, zType='Fixed')
    Heatmap = Amplitude_ConVar[:, MinFreq:MaxFreq]

    fig, ax = plt.subplots(figsize=(7,6))
    cax = fig.add_axes([0.95, 0.25, 0.04, 0.5])

    im = ax.imshow(Heatmap,  cmap='viridis', aspect='auto',interpolation='nearest') 
    ax.set(yticks=np.arange(0, N_Gen)[::20]);
    ax.set(xticks=np.arange(1, MaxFreq)[::5]-0.5, xticklabels=np.arange(1, MaxFreq)[::5]);
    ax.set_xlabel('Frequency', fontsize=16)
    ax.set_ylabel('Index of signals generated with a series of $\\acute{\\theta}$', fontsize=14) 

    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.show()

    
def VisReconGivenZ_CONA (ReconModel, ConData, LatDim, ZFix, N_Gen=300):
  

    zVal = np.tile(ZFix, (N_Gen,1))

    CON_Arange = GenConArangeSimple(ConData, N_Gen)
    SigGen = ReconModel([zVal, CON_Arange])

    # Create a colormap and normalize it based on the number of experiments
    cmap = cm.get_cmap('viridis')
    norm = plt.Normalize(0, N_Gen-1)
    FreqIDX = np.argmax(CON_Arange, axis=-1)
    norm2 = plt.Normalize(min(FreqIDX), max(FreqIDX))


    fig, ax = plt.subplots(figsize=(15, 7))
    for i in range(0, N_Gen):
        color = cmap(norm(i))
        ax.plot(SigGen[i], color=color)


    # Create color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm2)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Index of the maximum frequency ($\\acute{\\theta}$) for signal generation', size=13)

    plt.show()
    
    return SigGen
    
    
    
def VisReconGivenCON_ZA (ReconModel, LatDim, CON_Given,   N_Gen=300, MinZval = -3., MaxZval = 3., CutLower=-0.1, CutUpper = 0.1):
    
    zVal = np.linspace(MinZval, MaxZval, N_Gen*LatDim).reshape(N_Gen, -1)
    CON_Given = np.tile(CON_Given, (N_Gen,1))
        
    SigGen = ReconModel([zVal, CON_Given])
    
    # Create a colormap and normalize it based on the number of experiments
    cmap = cm.get_cmap('viridis')
    norm = plt.Normalize(0, N_Gen-1)
    norm2 = plt.Normalize(MinZval, MaxZval)


    fig, ax = plt.subplots(figsize=(15, 7))
    for i in range(0, N_Gen):
        color = cmap(norm(i))
        if np.min(zVal[i]) < CutLower or np.max(zVal[i]) > CutUpper:
            ax.plot(SigGen[i], color=color)


    # Create color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm2)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('z-value ranges')

    plt.show()
    
    return SigGen



def VisReconExtractCON_ZA (Data, idx,  ReconModel, CON_Given,  N_Gen=300):
    
    Sample = np.tile(Data[idx][None], (N_Gen, 1))
    RecPred = ReconModel([Sample, CON_Given])

    plt.figure(figsize=(15, 5))
    for i in range(N_Gen):
        plt.plot(RecPred[i])
    plt.plot(Data[idx],linestyle='--', color='black', label='True signal')
    plt.legend()
    
    return RecPred



