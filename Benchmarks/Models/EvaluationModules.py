import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import EntropyHub as EH


## Row and Column Permutation Entropy Index for TimeVae Model Benchmarking (RCPEI_BTI)
def RCPEI_TI (Model,  LatDim, N_Gen=300,   MinZval = -5., MaxZval = 5., N_FreqSel =3, MinFreq=1, MaxFreq=51 ):
   
    zValues = np.linspace(MinZval , MaxZval , N_Gen)
    ResList = []
    print(['Lat_ID', 'CWPE', 'RWPE', 'CWPE + RWPE'])
    for LatIdx in range(LatDim):
        zZeros = np.tile(np.zeros(LatDim), (N_Gen, 1))
        zZeros[:, LatIdx] = zValues

        ''' When given z latent values that have non-zero values in only one dimension, 
        generate signals of N_Gen size, then return the amplitude of the frequency through a Fourier transform. 2D Max[N_Gen, Zs.]'''
        SigGen = Model.decoder.predict(zZeros).reshape(N_Gen, -1)
        HalfLen = SigGen.shape[1]//2
        FFTRes = np.abs(np.fft.fft(SigGen, axis=-1)[:, :HalfLen]) 
        Amplitude = FFTRes/HalfLen
        Heatmap = Amplitude[:, MinFreq:MaxFreq]
        np.random.seed(0)
        Heatmap += np.random.normal(0., 1e-7, (Heatmap.shape))

        ''' Calculate column-wise permutation entropy.'''
        SortedIDX = np.argsort(Heatmap, axis=1)
        ListColWisePerEnt = []
        for IDSeq in SortedIDX.T:
            ListColWisePerEnt.append(np.maximum(EH.PermEn(IDSeq)[0][-1], 0.))
        MeanColWisePerEnt = np.mean(ListColWisePerEnt)    


        ''' Calculate N(i.e., N_FreqSel) row-wise permutation entropy.'''
        MaxIDX = np.argsort(np.mean(Heatmap, axis=0))[-N_FreqSel:]
        ListRowWisePerEnt = []
        for IDSeq in MaxIDX:
            ListRowWisePerEnt.append(np.maximum(EH.PermEn(Heatmap[:, IDSeq])[0][-1], 0.))
        MeanRowWisePerEnt = np.mean(ListRowWisePerEnt)


        'Aggregate results.'
        Res = [LatIdx, np.round(MeanColWisePerEnt, 4), np.round(MeanRowWisePerEnt, 4), np.round(MeanColWisePerEnt + MeanRowWisePerEnt, 4)]
        print(Res)
        ResList.append(Res)
    
    return ResList

## Row and Column Permutation Entropy Index for GP-HI Vae Model Benchmarking (RCPEI_GPHI)
def RCPEI_GPHI (Model,  LatDim, N_Gen=300,   MinZval = -5., MaxZval = 5., N_FreqSel =3, MinFreq=1, MaxFreq=51 ):
    
    assert Mode in ['GP','HI'], 'Mode only includes GP and HI'
    TimeN = Model.decoder.net.output.shape[1]
    zVal_Time = np.tile(np.linspace(MinZval, MaxZval, N_Gen)[:, None], TimeN)

    ResList = []
    print(['Lat_ID', 'Numerator', 'Denominator', 'VCSAE'])
    for LatIdx in range(LatDim):
        zValues = np.zeros((N_Gen,LatDim, TimeN))
        for zIdx in range(N_Gen):
            zValues[zIdx,LatIdx] = zVal_Time[zIdx]

        if Mode =='GP':
            pass
        elif Mode =='HI':
            zValues = np.transpose(zValues, (0,2,1))

        ''' When given z latent values that have non-zero values in only one dimension, 
        generate signals of N_Gen size, then return the amplitude of the frequency through a Fourier transform. 2D Max[N_Gen, Zs.]'''
        SigGen = Model.decode(zValues).mean().numpy().reshape(N_Gen, -1)

        HalfLen = SigGen.shape[1]//2
        FFTRes = np.abs(np.fft.fft(SigGen, axis=-1)[:, :HalfLen]) 
        Amplitude = FFTRes/HalfLen
        Heatmap = Amplitude[:, MinFreq:MaxFreq]
        np.random.seed(0)
        Heatmap += np.random.normal(0., 1e-7, (Heatmap.shape))

        ''' Calculate column-wise permutation entropy.'''
        SortedIDX = np.argsort(Heatmap, axis=1)
        ListColWisePerEnt = []
        for IDSeq in SortedIDX.T:
            ListColWisePerEnt.append(np.maximum(EH.PermEn(IDSeq)[0][-1], 0.))
        MeanColWisePerEnt = np.mean(ListColWisePerEnt)    

        ''' Calculate column-wise permutation entropy.'''
        SortedIDX = np.argsort(Heatmap, axis=1)
        ListColWisePerEnt = []
        for IDSeq in SortedIDX.T:
            ListColWisePerEnt.append(np.maximum(EH.PermEn(IDSeq)[0][-1], 0.))
        MeanColWisePerEnt = np.mean(ListColWisePerEnt)    

        "The ratio of the variance concentration of the axis with strong amplitude to entropy"
        SumHeatmap = np.sum(Heatmap, axis=0)
        StdHeatmap = np.std(Heatmap, axis=0)
        RateSumHeatmap = SumHeatmap / np.sum(SumHeatmap, axis=0)
        RateStdHeatmap = StdHeatmap / np.sum(StdHeatmap, axis=0)

        Numerator = np.mean(np.exp((RateSumHeatmap)**2) * np.exp((RateStdHeatmap)**2))
        Denominator = MeanRowWisePerEnt + MeanColWisePerEnt
        VCSAE =  Numerator / Denominator 

    return ResList


## The ratio of weighted power concentration to uncertainty (RWPCU)
def RWPCU_TI (Model,  LatDim, N_Gen=300,   MinZval = -5., MaxZval = 5., N_FreqSel =3, MinFreq=1, MaxFreq=51 ):
    
    '''The ratio of weighted power concentration to uncertainty'''
    zValues = np.linspace(MinZval , MaxZval , N_Gen)
    ResList = []
    print(['Lat_ID', 'Numerator', 'Denominator', 'VCSAE'])
    for LatIdx in range(LatDim):

        zZeros = np.tile(np.zeros(LatDim), (N_Gen, 1))
        zZeros[:, LatIdx] = zValues

        ''' When given z latent values that have non-zero values in only one dimension, 
        generate signals of N_Gen size, then return the amplitude of the frequency through a Fourier transform. 2D Max[N_Gen, Zs.]'''
        SigGen = Model.decoder.predict(zZeros).reshape(N_Gen, -1)
        HalfLen = SigGen.shape[1]//2
        FFTRes = np.abs(np.fft.fft(SigGen, axis=-1)[:, :HalfLen]) 
        Amplitude = FFTRes/HalfLen
        Heatmap = Amplitude[:, MinFreq:MaxFreq]
        np.random.seed(0)
        Heatmap += np.random.normal(0., 1e-7, (Heatmap.shape))

        ''' Calculate row-wise permutation entropy.'''
        ListRowWisePerEnt = []
        for Seq in Heatmap.T:
            ListRowWisePerEnt.append(np.maximum(EH.PermEn(Seq)[0][-1], 0.))
        RowWisePerEnt = np.array(ListRowWisePerEnt)

        ''' Calculate column-wise permutation entropy.'''
        Ranking = np.argsort(Heatmap, axis=1).argsort()
        ListColWisePerEnt = []
        for IDSeq in Ranking.T:
            ListColWisePerEnt.append(np.maximum(EH.PermEn(IDSeq)[0][-1], 0.))
        MeanColWisePerEnt = np.mean(ListColWisePerEnt)    

        "The ratio of the variance concentration of the axis with strong amplitude to entropy"
        SumHeatmap = np.sum(Heatmap, axis=0) # Total amplitude
        StdHeatmap = np.std(Heatmap, axis=0) # Total variation

        Weight = np.exp(-RowWisePerEnt)
        RateHeatmap = (SumHeatmap * Weight * StdHeatmap) / np.sum(SumHeatmap * Weight * StdHeatmap, axis=0)

        Numerator = np.sum(RateHeatmap**2)
        Denominator = MeanColWisePerEnt
        RWPCU =  Numerator / Denominator 

        'Aggregate results.'
        Res = [LatIdx, np.round(Numerator, 5), np.round(Denominator, 5), np.round(RWPCU, 5)]
        print(Res)
        ResList.append(Res)

    return ResList


## The ratio of the variance concentration of the axis with strong amplitude to entropy (VCSAE)
def RWPCU_GPHI (Model,  LatDim, Mode, N_Gen=300,   MinZval = -5., MaxZval = 5., N_FreqSel =3, MinFreq=1, MaxFreq=51 ):
    
    assert Mode in ['GP','HI'], 'Mode only includes GP and HI'
    TimeN = Model.decoder.net.output.shape[1]
    zVal_Time = np.tile(np.linspace(MinZval, MaxZval, N_Gen)[:, None], TimeN)

    ResList = []
    print(['Lat_ID', 'Numerator', 'Denominator', 'VCSAE'])
    for LatIdx in range(LatDim):
        zValues = np.zeros((N_Gen,LatDim, TimeN))
        for zIdx in range(N_Gen):
            zValues[zIdx,LatIdx] = zVal_Time[zIdx]

        if Mode =='GP':
            pass
        elif Mode =='HI':
            zValues = np.transpose(zValues, (0,2,1))


        ''' When given z latent values that have non-zero values in only one dimension, 
        generate signals of N_Gen size, then return the amplitude of the frequency through a Fourier transform. 2D Max[N_Gen, Zs.]'''
        SigGen = Model.decode(zValues).mean().numpy().reshape(N_Gen, -1)

        HalfLen = SigGen.shape[1]//2
        FFTRes = np.abs(np.fft.fft(SigGen, axis=-1)[:, :HalfLen]) 
        Amplitude = FFTRes/HalfLen
        Heatmap = Amplitude[:, MinFreq:MaxFreq]
        np.random.seed(0)
        Heatmap += np.random.normal(0., 1e-7, (Heatmap.shape))

        ''' Calculate row-wise permutation entropy.'''
        ListRowWisePerEnt = []
        for Seq in Heatmap.T:
            ListRowWisePerEnt.append(np.maximum(EH.PermEn(Seq)[0][-1], 0.))
        RowWisePerEnt = np.array(ListRowWisePerEnt)

        ''' Calculate column-wise permutation entropy.'''
        Ranking = np.argsort(Heatmap, axis=1).argsort()
        ListColWisePerEnt = []
        for IDSeq in Ranking.T:
            ListColWisePerEnt.append(np.maximum(EH.PermEn(IDSeq)[0][-1], 0.))
        MeanColWisePerEnt = np.mean(ListColWisePerEnt)    

        "The ratio of the variance concentration of the axis with strong amplitude to entropy"
        SumHeatmap = np.sum(Heatmap, axis=0) # Total amplitude
        StdHeatmap = np.std(Heatmap, axis=0) # Total variation

        Weight = np.exp(-RowWisePerEnt)
        RateHeatmap = (SumHeatmap * Weight * StdHeatmap) / np.sum(SumHeatmap * Weight * StdHeatmap, axis=0)

        Numerator = np.sum(RateHeatmap**2)
        Denominator = MeanColWisePerEnt
        RWPCU =  Numerator / Denominator 

        'Aggregate results.'
        Res = [LatIdx, np.round(Numerator, 5), np.round(Denominator, 5), np.round(RWPCU, 5)]
        print(Res)
        ResList.append(Res)

    return ResList


### Qualitative and Visual Evaluation
def HeatMap_TI (Model,  LatDim, ZFix, N_Gen=300, MinFreq=1, MaxFreq=51):

    zZeros = np.tile(np.zeros(LatDim), (N_Gen, 1))
    for KeyVal in ZFix.items():
        zZeros[:,KeyVal[0]] = KeyVal[1]
    
    ''' When given z latent values that have non-zero values in only one dimension, 
    generate signals of N_Gen size, then return the amplitude of the frequency through a Fourier transform. 2D Max[N_Gen, Zs.]'''
    SigGen = Model.decoder.predict(zZeros).reshape(N_Gen, -1)
    HalfLen = SigGen.shape[1]//2
    FFTRes = np.abs(np.fft.fft(SigGen, axis=-1)[:, :HalfLen]) 
    Amplitude = FFTRes/HalfLen
    Heatmap = Amplitude[:, MinFreq:MaxFreq]

    fig, ax = plt.subplots(figsize=(7,6))
    cax = fig.add_axes([0.95, 0.25, 0.04, 0.5])

    im = ax.imshow(Heatmap,  cmap='viridis', aspect='auto',interpolation='nearest') 
    ax.set(yticks=np.arange(1, N_Gen)[::10], yticklabels=np.round(KeyVal[1], 2)[::10]);
    ax.set(xticks=np.arange(1, MaxFreq)[::5]-0.5, xticklabels=np.arange(1, MaxFreq)[::5]);
    ax.set_xlabel('Frequency', fontsize=16)
    ax.set_ylabel('The z-values in a certain dimension', fontsize=16) 

    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.show()
    
    

def HeatMap_GPHI (Model, Mode, LatDim, ZFix, N_Gen=300, MinFreq=1, MaxFreq=51):
    
    assert Mode in ['GP','HI'], 'Mode only includes GP and HI'
    TimeN = Model.decoder.net.output.shape[1]
    zValues = np.zeros((N_Gen,LatDim, TimeN))
    for zIdx in range(N_Gen):
        for KeyVal in ZFix.items():
            zValues[zIdx,KeyVal[0]] = KeyVal[1][zIdx]
    
    if Mode =='GP':
        pass
    elif Mode =='HI':
        zValues = np.transpose(zValues, (0,2,1))
        
    ''' When given z latent values that have non-zero values in only one dimension, 
    generate signals of N_Gen size, then return the amplitude of the frequency through a Fourier transform. 2D Max[N_Gen, Zs.]'''
    SigGen = Model.decode(zValues).mean().numpy().reshape(N_Gen, -1)

    HalfLen = SigGen.shape[1]//2
    FFTRes = np.abs(np.fft.fft(SigGen, axis=-1)[:, :HalfLen]) 
    Amplitude = FFTRes/HalfLen
    Heatmap = Amplitude[:, MinFreq:MaxFreq]

    fig, ax = plt.subplots(figsize=(7,6))
    cax = fig.add_axes([0.95, 0.25, 0.04, 0.5])

    im = ax.imshow(Heatmap,  cmap='viridis', aspect='auto',interpolation='nearest') 
    ax.set(yticks=np.arange(1, N_Gen)[::10], yticklabels=np.round(KeyVal[1][:, 0], 2)[::10]);
    ax.set(xticks=np.arange(1, MaxFreq)[::5]-0.5, xticklabels=np.arange(1, MaxFreq)[::5]);
    ax.set_xlabel('Frequency', fontsize=16)
    ax.set_ylabel('The z-values in component dimensions', fontsize=16) 

    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.show()
    
    
    
def VisReconGivenZ_TI (Model, LatDim, ZFix,  N_Gen=300, MinFreqR=0, MaxFreqR=0.05):
    
    zZeros = np.tile(np.zeros(LatDim), (N_Gen, 1))
    for KeyVal in ZFix.items():
        zZeros[:,KeyVal[0]] = KeyVal[1]
    
    ''' When given z latent values that have non-zero values in only one dimension, 
    generate signals of N_Gen size, then return the amplitude of the frequency through a Fourier transform. 2D Max[N_Gen, Zs.]'''
    SigGen = Model.decoder.predict(zZeros).reshape(N_Gen, -1)
    
    # Create a colormap and normalize it based on the number of experiments
    cmap = cm.get_cmap('viridis')
    norm = plt.Normalize(0, N_Gen-1)
    norm2 = plt.Normalize(MinFreqR, MaxFreqR)

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
    cbar.set_label('Z values', size=14)

    plt.show()
    
    return SigGen


def VisReconGivenZ_GPHI (Model, Mode, LatDim, ZFix,  N_Gen=300, MinFreqR=0, MaxFreqR=0.05):
    
    assert Mode in ['GP','HI'], 'Mode only includes GP and HI'
    TimeN = Model.decoder.net.output.shape[1]
    zValues = np.zeros((N_Gen,LatDim, TimeN))
    for zIdx in range(N_Gen):
        for KeyVal in ZFix.items():
            zValues[zIdx,KeyVal[0]] = KeyVal[1][zIdx]
            
    if Mode =='GP':
        pass
    elif Mode =='HI':
        zValues = np.transpose(zValues, (0,2,1))
        
    ''' When given z latent values that have non-zero values in only one dimension, 
    generate signals of N_Gen size, then return the amplitude of the frequency through a Fourier transform. 2D Max[N_Gen, Zs.]'''
    SigGen = Model.decode(zValues).mean().numpy().reshape(N_Gen, -1)
    
    # Create a colormap and normalize it based on the number of experiments
    cmap = cm.get_cmap('viridis')
    norm = plt.Normalize(0, N_Gen-1)
    norm2 = plt.Normalize(MinFreqR, MaxFreqR)

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
    cbar.set_label('Z values', size=14)

    plt.show()
    
    return SigGen