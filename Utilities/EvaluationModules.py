import numpy as np
from scipy.stats import spearmanr, mode
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow.keras import Model
import itertools
import EntropyHub as EH

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import trange, tqdm




# Power spectral entropy 
def FFT_PSE (Data, ReducedAxis, MinFreq = 1, MaxFreq = 51):
    # Dimension check; this part operates with 3D tensors.
    # (Batch_size, N_sample, N_frequency)
    Data = Data[:,None] if len(Data.shape) < 3 else Data

    # Power Spectral Density
    HalfLen = Data.shape[-1]//2
    FFTRes = np.abs(np.fft.fft(Data, axis=-1)[..., :HalfLen])[..., MinFreq:MaxFreq]
    # (Batch_size, N_sample, N_frequency)
    PSD = (FFTRes**2)/FFTRes.shape[-1]

    # Probability Density Function
    if ReducedAxis == 'All':
        AggPSD = np.mean(PSD, axis=(0,1))
        # (N_frequency,)
        AggPSEPDF = AggPSD / np.sum(AggPSD, axis=(-1),keepdims=True)
    
    elif ReducedAxis =='Sample':
        AggPSD = np.mean(PSD, axis=(1))
        # (Batch_size, N_frequency)
        AggPSEPDF = AggPSD / np.sum(AggPSD, axis=(-1),keepdims=True)
    
    elif ReducedAxis == 'None':
        # (Batch_size, N_sample, N_frequency)
        AggPSEPDF = PSD / np.sum(PSD, axis=(-1),keepdims=True)    
        
    return AggPSEPDF


# Permutation entropy given PSE over each generation
def ProbPermutation(Data, Nframe=3, EpsProb = 1e-7):
    
    # Generate true permutation cases
    TruePerms = np.concatenate(list(itertools.permutations(np.arange(Nframe)))).reshape(-1, Nframe)

    # Get all permutation cases
    Data_Ext = tf.signal.frame(Data, frame_length=Nframe, frame_step=1, axis=-1)
    PermsTable =  np.argsort(Data_Ext, axis=-1)

    CountPerms = 1- (TruePerms[None,None,None] == PermsTable[:,:,:, None])
    CountPerms = 1-np.sum(CountPerms, axis=-1).astype('bool')
    CountPerms = np.sum(CountPerms, axis=(2))
    ProbCountPerms = CountPerms / np.sum(CountPerms, axis=-1, keepdims=True)
    
    return np.maximum(ProbCountPerms, EpsProb)    


# Searching for candidate Zj for plausible signal generation
def LocCandZs (BestZsMetrics, TrackerCandZ, Mode_Value, SumH, Samp_Z):
    
    for Freq, _ in BestZsMetrics.items():
        Mode_Idx = np.where(Mode_Value == Freq)[0]

        # Skipping the remainder of the code if there are no mode values present at the predefined frequencies.
        if len(Mode_Idx) <1: 
            continue;

        # Calculating the minimum of sum of H (Min_SumH) and Candidate Z-values(CandZs)
        Min_SumH_Idx = np.argmin(SumH[Mode_Idx])
        Min_SumH = np.min(SumH[Mode_Idx])
        CandZs = Samp_Z[[Mode_Idx[Min_SumH_Idx]]][0].flatten()
        CandZ_Idx = np.where(CandZs!=0)[0]
        
        #tracking results
        TrackerCandZ[Freq]['TrackZLOC'].append(CandZ_Idx[None])
        TrackerCandZ[Freq]['TrackZs'].append(CandZs[CandZ_Idx][None])
        TrackerCandZ[Freq]['TrackMetrics'].append(Min_SumH[None])

        # Updating the Min_SumH value if the current iteration value is smaller.
        if Min_SumH < BestZsMetrics[Freq][0]:
            BestZsMetrics[Freq] = [Min_SumH, CandZ_Idx, CandZs[CandZ_Idx]]
            print('Updated! ', 'Freq:', Freq, ', SumH_ZjFa:', np.round(Min_SumH, 4) , 
                  ' Z LOC:', CandZ_Idx, ' Z:', np.round(CandZs[CandZ_Idx], 4))
    
    return BestZsMetrics, TrackerCandZ


def MeanKLD(P,Q):
    return np.mean(np.sum(P*np.log(P/Q), axis=-1))


def Sampler (Data, SampModel,BatchSize=100):
    return SampModel.predict(Data, batch_size=BatchSize, verbose=1)   




# Conditional Mutual Information to evaluate model performance 
def CondMI (AnalData, SampModel, GenModel, FC_ArangeInp, SimSize = 1, NMiniBat=100, NGen=100, FcLimit=0.05, 
            MinFreq=1, MaxFreq=51, NSelZ = 1, FCmuEps = 0.05, ReparaStdZj = 10, PredBatchSize = 1000):
    
    # Parameters and values for the operation
    Ndata = len(AnalData)
    MASize = Ndata//NMiniBat
    LatDim = SampModel.output.shape[-1]
    NFCs = GenModel.get_layer('Inp_FCCommon').output.shape[-1] + GenModel.get_layer('Inp_FCEach').output.shape[-1]
    
    # Result tracking
    SubResDic = {'I_zE_Z':[],'I_zE_ZjZ':[],'I_zE_ZjFm':[],'I_zE_FaZj':[],'I_fcE_FmZj':[],'I_fcE_FaZj':[]}
    AggResDic = {'I_zE_Z':[],'I_zE_ZjZ':[],'I_zE_ZjFm':[],'I_zE_FaZj':[],'I_fcE_FmZj':[],'I_fcE_FaZj':[], 
                 'CMI_zE_ZjZ':[], 'CMI_zE_FcZj':[], 'CMI_fcE_FaFm':[]}
    BestZsMetrics = {i:[np.inf] for i in range(1, MaxFreq - MinFreq + 2)}
    TrackerCandZ = {i:{'TrackZLOC':[],'TrackZs':[],'TrackMetrics':[] } for i in range(1, MaxFreq - MinFreq + 2)} 
     
    
    ### monte carlo approximation
    I_zE_Z = 0
    I_zE_ZjZ = 0
    I_zE_ZjFm = 0
    I_zE_FaZj = 0
    I_fcE_FmZj = 0
    I_fcE_FaZj = 0
    
    
    
    # P(V=v)
    P_PSE = FFT_PSE(AnalData, 'All')

    with trange(MASize * SimSize , leave=False) as t:
        
        for sim in range(SimSize):

            SplitData = np.array_split(AnalData, MASize)
        
            for mini in range(MASize):
                print()

                '''
                ### Sampling Samp_Z and FCs and Reconstructing SigGen_ZFc ###

                - Shape of UniqSamp_Z: (NMiniBat, LatDim)
                - UniqSamp_Z ~ Q(z|y)

                - Samp_Z is a 3D tensor expanded by repeating the first axis (i.e., 0) of UniqSamp_Z by NGen times.
                - Shape of Samp_Z: (NMiniBat, NGen, LatDim) -> (NMiniBat*NGen, LatDim) for optimal use of GPU 
                - Shape of FCs: (NMiniBat, NGen, NFCs) -> (NMiniBat*NGen, LatDim) for optimal use of GPU 

                - SigGen_ZFc ~ Q(y | Samp_Z, FCs)
                - UniqSamp_Z ~ Q(z|y), FCs ~ Bern(fc, μ=0.5)

                '''
                # Sampling Samp_Z 
                UniqSamp_Z = Sampler(SplitData[mini], SampModel)
                Samp_Z =  np.broadcast_to(UniqSamp_Z[:, None], (NMiniBat, NGen, UniqSamp_Z.shape[-1])).reshape(-1, UniqSamp_Z.shape[-1])
                FCs = np.random.rand(NMiniBat *NGen, NFCs) * FcLimit


                '''
                ### Reconstructing SigGen_ZjFc ###

                - Masking is applied to select Samp_Zj from Samp_Z 
                  by assuming that the Samp_Z with indices other than j have a fixed mean value of '0' following a Gaussian distribution.

                - Shape of Samp_Zj: (NMiniBat, NGen, LatDim) -> (NMiniBat*NGen, LatDim) for optimal use of GPU 
                - Shape of FCs: (NMiniBat, NGen, NFCs) -> (NMiniBat*NGen, LatDim) for optimal use of GPU 

                - SigGen_ZjFc ~ Q(y | Samp_Zj, FCs)
                - Samp_Zj ~ Q(z|y), j∼U(1,LatDim), FCs ~ Bern(fc, μ=0.5)

                '''

                # Masking for selecting Samp_Zj from Samp_Z 
                Mask_Z = np.zeros((NMiniBat*NGen, LatDim))
                for i in range(NMiniBat*NGen):
                    Mask_Z[i, np.random.choice(LatDim, NSelZ,replace=False )] = 1
                Samp_Zj = Samp_Z * Mask_Z



                '''
                ### Reconstructing SigGen_ZjFcRPT ###

                - Samp_ZjRPT is sampled from a Gaussian distribution with a mean of 0 and standard deviation; Samp_ZjRPT ~ N(0, ReparaStdZj)
                  by assuming that the Samp_Z with indices other than j have a fixed mean value of '0', 
                  then it's repeated NGen times along the second axis (i.e., 1).

                - Shape of Samp_ZjRPT: (NMiniBat, NGen, LatDim) -> (NMiniBat*NGen, LatDim) for optimal use of GPU 
                - Shape of FCs: (NMiniBat, NGen, NFCs) -> (NMiniBat*NGen, LatDim) for optimal use of GPU 

                - SigGen_ZjFcRPT ~ Q(y | Samp_ZjRPT, FCs)
                - Samp_ZjRPT ~ N(0, ReparaStdZj), j∼U(1,LatDim), FCs ~ Bern(fc, μ=0.5)
                - In the expression j∼U(1,LatDim), j corresponds to LatDim and all js are selected randomly.

                '''
                

                # Selecting Samp_Zj from Guassian dist.
                Samp_ZjRPT = []
                for i in range(NMiniBat):
                    Mask_Z = np.zeros((LatDim))
                    
                    # LatDim-wise Z sampling
                    Mask_Z[ np.random.choice(LatDim, NSelZ,replace=False )]= np.random.normal(0, ReparaStdZj, NSelZ)

                    # Setting the same Z value within the N generated signals (NGen).
                    Samp_ZjRPT.append(np.broadcast_to(Mask_Z[None], (NGen,LatDim))[None]) 

                Samp_ZjRPT = np.concatenate(Samp_ZjRPT).reshape(NMiniBat *NGen, LatDim)


                '''
                ### Reconstructing SigGen_ZjFcAr ###

                - FC_Arange: The FC values are generated NGen times, based on the linspace with a fixed interval (0 ~ FcLimit).
                  Thus, each sample(i.e., NMiniBat) has FC values that are sorted and generated by the linspace (i.e., NGen).

                - Shape of Samp_ZjRPT: (NMiniBat, NGen, LatDim) -> (NMiniBat*NGen, LatDim) for optimal use of GPU 
                - Shape of FC_Arange: (NMiniBat, NGen, NFCs) -> (NMiniBat*NGen, LatDim) for optimal use of GPU 

                - SigGen_ZjFcAr ~ Q(y | Samp_ZjRPT, FC_Arange)
                - Samp_ZjRPT ~ N(0, ReparaStdZj), j∼U(1,LatDim), FC_Arange∼U(0,FcLimit)
                - j corresponds to LatDim and all js are selected randomly.

                '''

                # The FC values are generated NGen times, based on the linspace with a fixed interval.
                FC_Arange = np.broadcast_to(FC_ArangeInp[None], (NMiniBat, NGen, NFCs)).reshape(-1, NFCs)



                '''
                ### Reconstructing SigGen_ZjFcMu ###

                - Given that FC ~ (fc, μ=0.5), the expected value of FCmu should be 0.5 x FcLimit.
                  This would result in all values within a single sample (NMiniBat, NGen) being equal.
                  Thus, we assumed FC_Rand as FC_μ x FcLimit + eps, where FC_μ=0.5 and eps ~ N(0, 0.5*FcLimit*FCmuEps)

                - Shape of Samp_ZjRPT: (NMiniBat, NGen, LatDim) -> (NMiniBat*NGen, LatDim) for optimal use of GPU 
                - Shape of FC_Rand: (NMiniBat, NGen, NFCs) -> (NMiniBat*NGen, LatDim) for optimal use of GPU 

                - SigGen_ZjFcMu ~ Q(y | Samp_ZjRPT, FC_Rand)
                - Samp_ZjRPT ~ N(0, ReparaStdZj), j∼U(1,LatDim), FC_Arange∼U(0,FcLimit)
                - j corresponds to LatDim and all js are selected randomly.

                '''

                # FC_Rand are sampled as FC_μ x FcLimit + eps, where FC_μ=0.5 and eps ~ N(0, 0.5*FcLimit*FCmuEps)
                FC_Rand = np.zeros_like(FCs) + FcLimit * 0.5 + np.random.normal(0, (FcLimit * 0.5)*FCmuEps, (FCs.shape))



                '''
                ### Signal reconstruction ###

                - To maximize the efficiency of GPU utilization, 
                  we performed a binding operation on (NMiniBat, NGen, LatDim) for Zs and (NMiniBat, NGen, NFCs) for FCs, respectively, 
                  transforming them to (NMiniBat * NGen, LatDim) and (NMiniBat * NGen, NFCs). 
                  After the computation, we then reverted them back to their original dimensions.

                '''

                Set_FCs = np.concatenate([FCs,FCs,FCs,FC_Arange,FC_Rand])
                Set_Zs = np.concatenate([Samp_Z,Samp_Zj,Samp_ZjRPT,Samp_ZjRPT,Samp_ZjRPT])
                Set_Pred = GenModel.predict([Set_FCs[:, :2], Set_FCs[:, 2:], Set_Zs], batch_size=PredBatchSize, verbose=1).reshape(-1, NMiniBat, NGen, AnalData.shape[-1])
                SigGen_ZFc, SigGen_ZjFc, SigGen_ZjFcRPT, SigGen_ZjFcAr, SigGen_ZjFcMu = [np.squeeze(SubPred) for SubPred in np.split(Set_Pred, 5) ]  



                # Cumulative Power Spectral Entropy (PSE) over each frequency
                Q_PSE_ZFc = FFT_PSE(SigGen_ZFc, 'Sample')
                Q_PSE_ZjFc = FFT_PSE(SigGen_ZjFc, 'Sample')

                Q_PSE_ZjFcRPT = FFT_PSE(SigGen_ZjFcRPT, 'Sample')
                Q_PSE_ZjFcAr = FFT_PSE(SigGen_ZjFcAr, 'Sample')
                Q_PSE_ZjFcMu = FFT_PSE(SigGen_ZjFcMu, 'Sample')

                SubPSE_ZjFcRPT = FFT_PSE(SigGen_ZjFcRPT, 'None').transpose(0,2,1)
                SubPSE_ZjFcMu = FFT_PSE(SigGen_ZjFcMu, 'None').transpose(0,2,1)
                SubPSE_ZjFcAr = FFT_PSE(SigGen_ZjFcAr, 'None').transpose(0,2,1)


                # Permutation Entropy given PSE over each generation
                Q_FcPSE_ZjFcRPT = ProbPermutation(SubPSE_ZjFcRPT, Nframe=3, EpsProb = 1e-7)
                Q_FcPSE_ZjFcMu = ProbPermutation(SubPSE_ZjFcMu, Nframe=3, EpsProb = 1e-7)
                Q_FcPSE_ZjFcAr = ProbPermutation(SubPSE_ZjFcAr, Nframe=3, EpsProb = 1e-7)


                # Conditional mutual information; zE and fcE stand for z-wise power spectral entropy and fc-wise permutation entropy, respectively.
                I_zE_Z_ = MeanKLD(Q_PSE_ZFc, P_PSE[None] ) # I(zE;Z)
                I_zE_ZjZ_ = MeanKLD(Q_PSE_ZjFc, Q_PSE_ZFc )  # I(zE;Zj|Z)
                I_zE_ZjFm_ =  MeanKLD(Q_PSE_ZjFcMu, P_PSE[None] ) # I(zE;Zj)
                I_zE_FaZj_ = MeanKLD(Q_PSE_ZjFcAr, Q_PSE_ZjFcMu ) # I(zE;FC|Zj)
                I_fcE_FmZj_ = MeanKLD(Q_FcPSE_ZjFcMu, Q_FcPSE_ZjFcRPT) # I(fcE;Zj)
                I_fcE_FaZj_ = MeanKLD(Q_FcPSE_ZjFcAr, Q_FcPSE_ZjFcMu) # I(fcE;FC|Zj)


                print('I_zE_Z :', I_zE_Z_)
                SubResDic['I_zE_Z'].append(I_zE_Z_)
                I_zE_Z += I_zE_Z_

                print('I_zE_ZjZ :', I_zE_ZjZ_)
                SubResDic['I_zE_ZjZ'].append(I_zE_ZjZ_)
                I_zE_ZjZ += I_zE_ZjZ_

                print('I_zE_ZjFm :', I_zE_ZjFm_)
                SubResDic['I_zE_ZjFm'].append(I_zE_ZjFm_)
                I_zE_ZjFm += I_zE_ZjFm_

                print('I_zE_FaZj :', I_zE_FaZj_)
                SubResDic['I_zE_FaZj'].append(I_zE_FaZj_)
                I_zE_FaZj += I_zE_FaZj_

                print('I_fcE_FmZj :', I_fcE_FmZj_)
                SubResDic['I_fcE_FmZj'].append(I_fcE_FmZj_)
                I_fcE_FmZj += I_fcE_FmZj_

                print('I_fcE_FaZj :', I_fcE_FaZj_)
                SubResDic['I_fcE_FaZj'].append(I_fcE_FaZj_)
                I_fcE_FaZj += I_fcE_FaZj_


                # Locating the candidate Z values that generate plausible signals.
                H_zE_ZjFa = -np.sum(Q_PSE_ZjFcAr * np.log(Q_PSE_ZjFcAr), axis=-1)
                H_fcE_ZjFa = np.mean(-np.sum(Q_FcPSE_ZjFcAr * np.log(Q_FcPSE_ZjFcAr), axis=-1), axis=-1)
                SumH_ZjFa = H_zE_ZjFa + H_fcE_ZjFa


                # Calculating mode values of SigGen_ZjFcAr
                Q_PSE_ZjFcAr_3D = FFT_PSE(SigGen_ZjFcAr, 'None')
                # The 0 frequency is excluded as it represents the constant term; by adding 1 to the index, the frequency and index can be aligned to be the same.
                Max_Value_Label = np.argmax(Q_PSE_ZjFcAr_3D, axis=-1) + 1
                Mode_Value = mode(Max_Value_Label.T, axis=0, keepdims=False)[0]
                UniqSamp_ZjRPT = Samp_ZjRPT.reshape(NMiniBat, NGen, -1)[:, 0]
                BestZsMetrics, TrackerCandZ = LocCandZs (BestZsMetrics, TrackerCandZ, Mode_Value, SumH_ZjFa, UniqSamp_ZjRPT)

                t.update(1)

        # CMI(V;Zj, Z)
        I_zE_Z /= (MASize*SimSize)
        AggResDic['I_zE_Z'].append(I_zE_Z)
        I_zE_ZjZ /= (MASize*SimSize)
        AggResDic['I_zE_ZjZ'].append(I_zE_ZjZ)
        CMI_zE_ZjZ = I_zE_Z + I_zE_ZjZ             
        AggResDic['CMI_zE_ZjZ'].append(CMI_zE_ZjZ)

        # CMI(V;FC,Zj)
        I_zE_ZjFm /= (MASize*SimSize)
        AggResDic['I_zE_ZjFm'].append(I_zE_ZjFm)
        I_zE_FaZj /= (MASize*SimSize)
        AggResDic['I_zE_FaZj'].append(I_zE_FaZj)
        CMI_zE_FcZj = I_zE_ZjFm + I_zE_FaZj       
        AggResDic['CMI_zE_FcZj'].append(CMI_zE_FcZj)

        # CMI(VE;FA,FM)
        I_fcE_FmZj /= (MASize*SimSize)
        AggResDic['I_fcE_FmZj'].append(I_fcE_FmZj)
        I_fcE_FaZj /= (MASize*SimSize)
        AggResDic['I_fcE_FaZj'].append(I_fcE_FaZj)
        CMI_fcE_FaFm = I_fcE_FmZj + I_fcE_FaZj    
        AggResDic['CMI_fcE_FaFm'].append(CMI_fcE_FaFm)
    
    
    return AggResDic, SubResDic, (BestZsMetrics,TrackerCandZ)





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
    cbar.set_label('Frequency given for generating signals', size=14)

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



