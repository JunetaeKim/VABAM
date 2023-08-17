import numpy as np
from scipy.stats import mode
import itertools
from tqdm import trange, tqdm

import tensorflow as tf
from tensorflow.keras import Model

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Utilities.AncillaryFunctions import FFT_PSD, ProbPermutation, MeanKLD, Sampler, SamplingZ, SamplingZj


       
class Evaluator ():
    
    def __init__ (self, MinFreq=1, MaxFreq=51,  SimSize = 1, NMiniBat=100, 
                  NGen=100, ReparaStdZj = 1, NSelZ = 1, SampBatchSize = 1000, GenBatchSize = 1000, GPU=True):
        
        # Optional parameters with default values
        self.MinFreq = MinFreq               # The minimum frequency value within the analysis range (default = 1).
        self.MaxFreq = MaxFreq               # The maximum frequency value within the analysis range (default = 51).
        self.SimSize = SimSize               # Then umber of simulation repetitions for aggregating metrics (default: 1)
        self.NMiniBat = NMiniBat             # The size of the mini-batch, splitting the task into N pieces of size NMiniBat.
        self.NGen = NGen                     # The number of generations (i.e., samplings) within the mini-batch.
        self.ReparaStdZj = ReparaStdZj       # The size of the standard deviation when sampling Zj (Samp_ZjRPT ~ N(0, ReparaStdZj)).
        self.NSelZ = NSelZ                   # The size of js to be selected at the same time when selecting Zj (default: 1).
        self.SampBatchSize = SampBatchSize   # The batch size during prediction of the sampling model.
        self.GenBatchSize= GenBatchSize      # The batch size during prediction of the generation model.
        self.GPU = GPU                       # GPU vs CPU during model predictions (i.e., for SampModel and GenModel).
        
        
        
    
    ''' ------------------------------------------------------ Ancillary Functions ------------------------------------------------------'''

    ### ----------- Searching for candidate Zj for plausible signal generation ----------- ###
    def LocCandZs (self, Mode_Value, SumH, Samp_Z,):
        
        for Freq, _ in self.BestZsMetrics.items():
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
            self.TrackerCandZ_Temp[Freq]['TrackZLOC'].append(CandZ_Idx[None])
            self.TrackerCandZ_Temp[Freq]['TrackZs'].append(CandZs[CandZ_Idx][None])
            self.TrackerCandZ_Temp[Freq]['TrackMetrics'].append(Min_SumH[None])

            # Updating the Min_SumH value if the current iteration value is smaller.
            if Min_SumH < self.BestZsMetrics[Freq][0]:
                self.BestZsMetrics[Freq] = [Min_SumH, CandZ_Idx, CandZs[CandZ_Idx]]
                print('Candidate Z updated! ', 'Freq:', Freq, ', SumH_ZjFa:', np.round(Min_SumH, 4) , 
                      ' Z LOC:', CandZ_Idx, ' Z:', np.round(CandZs[CandZ_Idx], 4))
        

    
    ### ------------------------  Selecting nested Z-LOC and Z values --------------------- ###
    def SubNestedZFix(self, SubTrackerCandZ , NSelZ=None):
        
        # Setting arguments
        NSelZ = self.NSelZ if NSelZ is None else NSelZ
        
        Cnt = itertools.count()
        Results = {next(Cnt):{ TrackZLOC[i] : TrackZs[i] for i in range(NSelZ)} 
                    for TrackZLOC,TrackZs, TrackMetrics 
                    in zip(SubTrackerCandZ['TrackZLOC'], SubTrackerCandZ['TrackZs'], SubTrackerCandZ['TrackMetrics'])
                    if TrackMetrics < self.MetricCut }
        return Results
    
    
    
    ### ------------------------------ Conducting task iteration ------------------------------ ###
    def Iteration (self, TaskLogic):

        # Just functional code for setting the initial position of the progress bar 
        self.StartBarPoint = self.TotalIterSize*(self.iter/self.TotalIterSize) 
        with trange(self.iter, self.TotalIterSize , initial=self.StartBarPoint, leave=False) as t:

            for sim in range(self.sim, self.SimSize):
                self.sim = sim

                # Check the types of ancillary data fed into the sampler model and define the pipeline accordingly.
                if self.SecDataType == 'CON' : 
                    SplitData = [np.array_split(sub, self.SubIterSize) for sub in self.AnalData]   

                else: # For models with a single input such as VAE and TCVAE.
                    SplitData = np.array_split(self.AnalData, self.SubIterSize)    

                for mini in range(self.mini, self.SubIterSize):
                    self.mini = mini
                    self.iter += 1
                    print()

                    # Core part; the task logic as the function
                    if self.SecDataType  == 'CON' : 
                        TaskLogic([subs[mini] for subs in SplitData])
                    else:
                        TaskLogic(SplitData[mini])

                    t.update(1)
    
    
    
    ### ------------------- Selecting post-sampled Z values for generating plausible signals ------------------- ###
    def SelPostSamp_Zj (self, MetricCut=np.inf, BestZsMetrics=None, TrackerCandZ=None, NSelZ=None, SavePath=None ):
        
        ## Optional parameters
        # MetricCut: The threshold value for selecting Zs whose Entropy of PSD (i.e., SumH) is less than the MetricCut
        self.MetricCut = MetricCut

        # Setting arguments
        BestZsMetrics = self.BestZsMetrics if BestZsMetrics is None else BestZsMetrics
        TrackerCandZ = self.TrackerCandZ if TrackerCandZ is None else TrackerCandZ
        NSelZ = self.NSelZ if NSelZ is None else NSelZ
                
        # Exploring FreqIDs available for signal generation  
        self.CandFreqIDs = [item[0] for item in BestZsMetrics.items() if (item[1][0] != np.inf) and (item[1][0] < self.MetricCut)]

        # Selecting nested Z-LOC and Z values
        self.NestedZFix = {FreqID : self.SubNestedZFix(TrackerCandZ[FreqID], ) for FreqID in self.CandFreqIDs}

        # Creating a tensor of Z values for signal generation
        PostSamp_Zj = []
        for SubZFix in self.NestedZFix.items():
            for item in SubZFix[1].items():
                Mask_Z = np.zeros((self.LatDim))
                Mask_Z[list(item[1].keys())] =  list(item[1].values())
                PostSamp_Zj.append(Mask_Z[None])
        self.PostSamp_Zj = np.concatenate(PostSamp_Zj, axis=0)
        
        
        # Counting the number of obs in NestedZs
        NPostZs =0 
        for item in self.NestedZFix.items():
            NPostZs += len(item[1])

        print('The total number of sets in NestedZs:', NPostZs)

        
        # Saving intermedicate results into the hard disk
        if SavePath is not None:
            np.save(SavePath, PostSamp_Zj) # Save data

        return self.PostSamp_Zj, self.NestedZFix
    
    
        
    ### -------------- Evaluating the KLD between the PSD of the true signals and the generated signals ---------------- ###
    def KLD_TrueGen (self, RepeatSize=1, PostSamp_Zj=None, FcLimit=None, PlotDist=True, SecDataType=None):
    
        # GPU vs CPU
        def CompResource (Data): 

            if self.GPU==False:
                with tf.device('/CPU:0'):
                    PredVal = self.GenModel.predict(Data, batch_size=self.GenBatchSize, verbose=1)
            else:
                PredVal = self.GenModel.predict(Data, batch_size=self.GenBatchSize, verbose=1)

            return PredVal


        ## Optional parameters
        # RepeatSize: The number of iterations to repetitively generate identical PostSamp_Zj; 
                    # this is to observe variations in other inputs such as FCs while PostSamp_Zj remains constant.
        # SecDataType: Secondary data type; Use 'FCR' for FC values chosen randomly, 'FCA' for FC values given by arrange, 
                    # and 'CON' for conditional inputs such as power spectral density.

    
        # Setting arguments
        PostSamp_Zj = self.PostSamp_Zj if PostSamp_Zj is None else PostSamp_Zj
        SecDataType = self.SecDataType if SecDataType is None else SecDataType
        if FcLimit is not None:
            FcLimit = self.FcLimit if FcLimit is None else FcLimit

        # Repeating PostSamp_Zj RepeatSize times.
        Ext_Samp_Zj = np.tile(PostSamp_Zj[:, None], (1, RepeatSize, 1))
        NSamp, NVar = Ext_Samp_Zj.shape[0], Ext_Samp_Zj.shape[1]
        Ext_Samp_Zj = np.reshape(Ext_Samp_Zj, (-1, self.LatDim))
        
        
        if SecDataType == 'FCR': # FC random
            Ext_Samp_FCs = np.random.rand(NSamp, NVar, self.NFCs) * self.FcLimit
            Ext_Samp_FCs = np.reshape(Ext_Samp_FCs, (-1, self.NFCs))
            Data = [Ext_Samp_FCs[:, :2], Ext_Samp_FCs[:, 2:], Ext_Samp_Zj]
            
        elif SecDataType == 'FCA': # FC arrange
            Ext_Samp_FCs = np.tile(np.linspace(1e-7, self.FcLimit, RepeatSize)[None, :, None], (NSamp, 1, self.NFCs))
            Ext_Samp_FCs = np.reshape(Ext_Samp_FCs, (-1, self.NFCs))
            Data = [Ext_Samp_FCs[:, :2], Ext_Samp_FCs[:, 2:], Ext_Samp_Zj]
            
        elif SecDataType == 'CON': # Conditional inputs such as power spectral density
            RandIdx = np.random.permutation(len(Ext_Samp_Zj))
            Data = [Ext_Samp_Zj, self.AnalData[1][RandIdx]]
        
        else:
            Data = Ext_Samp_Zj


        self.GenSamp = CompResource (Data)
            

        # Calculating the KLD between the PSD of the true signals and the generated signals    
        PSDGenSamp =  FFT_PSD(self.GenSamp, 'All', MinFreq = 1, MaxFreq = 51)
        if SecDataType == 'CON': # Conditional inputs such as power spectral density
            PSDTrueData =  FFT_PSD(self.AnalData[0], 'All', MinFreq = 1, MaxFreq = 51)
        else:
            PSDTrueData =  FFT_PSD(self.AnalData, 'All', MinFreq = 1, MaxFreq = 51)
            
        self.KldPSD_GenTrue = MeanKLD(PSDGenSamp, PSDTrueData)
        self.KldPSD_TrueGen  = MeanKLD(PSDTrueData, PSDGenSamp)
        self.MeanKld_GTTG = (self.KldPSD_GenTrue + self.KldPSD_TrueGen) / 2
        

        print(self.KldPSD_GenTrue)
        print(self.KldPSD_TrueGen)
        print(self.MeanKld_GTTG)

        if PlotDist==True:
            plt.plot(PSDGenSamp, c='green', label='Generated')
            plt.plot(PSDTrueData, c='orange', label='True')
            plt.fill_between(np.arange(len(PSDTrueData)), PSDTrueData, color='orange', alpha=0.5)
            plt.fill_between(np.arange(len(PSDGenSamp)), PSDGenSamp, color='green', alpha=0.5)
            plt.legend()    
    
    
        
    
    
    
    ''' ------------------------------------------------------ Main Functions ------------------------------------------------------'''
    
    ### -------------------------- Evaluating the performance of the model using both Z and FC inputs  -------------------------- ###
    def Eval_ZFC (self, AnalData, SampModel, GenModel, FC_ArangeInp, FcLimit=0.05,  WindowSize=3, Continue=True, SampZType='Model', SecDataType='FCA'):
        
        ## Required parameters
        self.AnalData = AnalData             # The data to be used for analysis.
        self.SampModel = SampModel           # The model that samples Zs.
        self.GenModel = GenModel             # The model that generates signals based on given Zs and FCs.
        self.FC_ArangeInp = FC_ArangeInp     # The 2D matrix (N_sample, NFCs) containing FCs values that the user creates and inputs directly.
        self.SecDataType = SecDataType       # The ancillary data-type: Use 'FCR' for FC values chosen randomly, 'FCA' for FC values given by arrange, 
                                             # and 'CON' for conditional inputs such as power spectral density.
        assert SecDataType in ['FCA','FCR','CON'], "Please verify the value of 'SecDataType'. Only 'FCA', 'FCR', or 'CON' are valid."
        
        
        ## Optional parameters with default values ##
        # WindowSize: The window size when calculating permutation entropy (default: 3)
        # Continue: Start from the beginning (Continue = False) vs. Continue where left off (Continue = True)
        self.SampZType = SampZType  # Z~ N(Zμ|y, σ) (SampZType = 'Model') vs. Z ~ N(0, ReparaStdZj) (SampZType = 'Random')
        self.FcLimit = FcLimit # The threshold value of the max of the FC value input into the generation model (default: 0.05, i.e., frequency 5 Hertz)      
            
        ## Intermediate variables
        self.Ndata = len(AnalData) # The dimension size of the data.
        self.NFCs = GenModel.get_layer('Inp_FCEach').output.shape[-1] + GenModel.get_layer('Inp_FCCommon').output.shape[-1] # The dimension size of FCs.
        self.LatDim = SampModel.output.shape[-1] # The dimension size of Z.
        self.SigDim = AnalData.shape[-1] # The dimension (i.e., length) size of the raw signal.
        self.SubIterSize = self.Ndata//self.NMiniBat
        self.TotalIterSize = self.SubIterSize * self.SimSize
        
        
        # Functional trackers
        if Continue == False or not hasattr(self, 'iter'):
            self.sim, self.mini, self.iter = 0, 0, 0
        
            ## Result trackers
            self.SubResDic = {'I_zPSD_Z':[],'I_zPSD_ZjZ':[],'I_zPSD_ZjFc':[],'I_zPSD_FaZj':[],'I_fcPE_FcZj':[],'I_fcPE_FaZj':[]}
            self.AggResDic = {'I_zPSD_Z':[],'I_zPSD_ZjZ':[],'I_zPSD_ZjFc':[],'I_zPSD_FaZj':[],'I_fcPE_FcZj':[],'I_fcPE_FaZj':[], 
                         'CMI_zPSD_ZjZ':[], 'CMI_zPSD_FcZj':[], 'CMI_fcPE_FaFc':[]}
            self.BestZsMetrics = {i:[np.inf] for i in range(1, self.MaxFreq - self.MinFreq + 2)}
            self.TrackerCandZ_Temp = {i:{'TrackZLOC':[],'TrackZs':[],'TrackMetrics':[] } for i in range(1, self.MaxFreq - self.MinFreq + 2)} 
            self.I_zPSD_Z, self.I_zPSD_ZjZ, self.I_zPSD_ZjFc, self.I_zPSD_FaZj, self.I_fcPE_FcZj, self.I_fcPE_FaZj = 0,0,0,0,0,0
        
        
        
        ### ------------------------------------------------ Task logics ------------------------------------------------ ###
        
        # P(V=v)
        ## Data shape: (N_frequency)
        self.P_PSPDF = FFT_PSD(self.AnalData, 'All', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)
        
        
        def TaskLogic(SubData):

            ### ------------------------------------------------ Sampling ------------------------------------------------ ###
            # Updating NMiniBat; If there is a remainder in Ndata/NMiniBat, NMiniBat must be updated." 
            self.NMiniBat = len(SubData) 

            # Sampling Samp_Z 
            self.Samp_Z = SamplingZ(SubData, self.SampModel, self.NMiniBat, self.NGen, 
                               BatchSize = self.SampBatchSize, GPU=self.GPU, SampZType=self.SampZType, ReparaStdZj=self.ReparaStdZj)

            # Selecting Samp_Zj from Samp_Z 
            ## For Samp_Zj, j is selected randomly across both the j and generation axes.
            self.Samp_Zj = SamplingZj (self.Samp_Z, self.NMiniBat, self.NGen, self.LatDim, self.NSelZ, Axis=1)
            ## For Samp_ZjRPT, the same j is selected in all generations within a mini-batch.
            self.Samp_ZjRPT = SamplingZj (self.Samp_Z, self.NMiniBat, self.NGen, self.LatDim, self.NSelZ, Axis=2)



            # Sampling FCs
            ## Shape of FCs: (NMiniBat*NGen, NFCs) instead of (NMiniBat, NGen, NFCs) for optimal use of GPU
            self.FCs = np.random.rand(self.NMiniBat * self.NGen, self.NFCs) * FcLimit

            # Generating FCμ: 
            self.FCmu = np.zeros_like(self.FCs) + FcLimit * 0.5

            # Generating FC values with a fixed interval that increases at equal increments.
            self.FC_Arange = np.broadcast_to(self.FC_ArangeInp[None], (self.NMiniBat, self.NGen, self.NFCs)).reshape(-1, self.NFCs)




            ### ------------------------------------------------ Signal reconstruction ------------------------------------------------ ###
            '''
            - To maximize the efficiency of GPU utilization, 
              we performed a binding operation on (NMiniBat, NGen, LatDim) for Zs and (NMiniBat, NGen, NFCs) for FCs, respectively, 
              transforming them to (NMiniBat * NGen, LatDim) and (NMiniBat * NGen, NFCs). 
              After the computation, we then reverted them back to their original dimensions.

            '''
            ## Binding the samples together, generate signals through the model 
            Set_FCs = np.concatenate([self.FCmu,   self.FCmu,     self.FCs,         self.FC_Arange])
            Set_Zs = np.concatenate([self.Samp_Z,  self.Samp_Zj,  self.Samp_ZjRPT,  self.Samp_ZjRPT])


            # Choosing GPU or CPU and generating signals
            if self.GPU==False:
                with tf.device('/CPU:0'):
                    Set_Pred = self.GenModel.predict([Set_FCs[:, :2], Set_FCs[:, 2:], Set_Zs], batch_size=self.GenBatchSize, verbose=1)                            
            else:
                Set_Pred = self.GenModel.predict([Set_FCs[:, :2], Set_FCs[:, 2:], Set_Zs], batch_size=self.GenBatchSize, verbose=1)


            # Re-splitting predictions for each case
            Set_Pred = Set_Pred.reshape(-1, self.NMiniBat, self.NGen, self.SigDim)
            self.SigGen_Z, self.SigGen_Zj, self.SigGen_ZjRptFC, self.SigGen_ZjRptFCar = [np.squeeze(SubPred) for SubPred in np.split(Set_Pred, 4)]  



            ### ---------------------------- Cumulative Power Spectral Density (PSD) over each frequency -------------------------------- ###
            # Return shape: (Batch_size, N_frequency)
            self.Q_PSPDF_Z = FFT_PSD(self.SigGen_Z, 'Sample', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)
            self.Q_PSPDF_Zj = FFT_PSD(self.SigGen_Zj, 'Sample', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)
            self.Q_PSPDF_ZjRptFC = FFT_PSD(self.SigGen_ZjRptFC, 'Sample', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)
            self.Q_PSPDF_ZjRptFCar = FFT_PSD(self.SigGen_ZjRptFCar, 'Sample', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)

            # Return shape: (Batch_size, N_frequency, N_sample)
            self.SubPSPDF_ZjRptFC = FFT_PSD(self.SigGen_ZjRptFC, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).transpose(0,2,1)
            self.SubPSPDF_ZjRptFCar = FFT_PSD(self.SigGen_ZjRptFCar, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).transpose(0,2,1)

            # Return shape: (Batch_size, 1, N_frequency)
            self.SubPSPDF_Batch = FFT_PSD(SubData[:,None], 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)
            self.SubPSPDF_Batch.sort(0)


            ### ---------------------------- Permutation Entropy given PSD over each generation -------------------------------- ###
            # Return shape: (Batch_size, N_frequency, N_permutation_cases)
            self.Q_PEPDF_ZjRptFC = ProbPermutation(self.SubPSPDF_ZjRptFC, WindowSize=WindowSize)
            self.Q_PEPDF_ZjRptFCar = ProbPermutation(self.SubPSPDF_ZjRptFCar, WindowSize=WindowSize)
            self.Q_PEPDF_Batch = ProbPermutation(self.SubPSPDF_Batch, WindowSize=WindowSize)


            ### ---------------------------------------- Conditional mutual information ---------------------------------------- ###
            # zPSD and fcPE stand for z-wise power spectral density and fc-wise permutation entropy, respectively.
            I_zPSD_Z_ = MeanKLD(self.Q_PSPDF_Z, self.P_PSPDF[None] ) # I(zPSD;Z)
            I_zPSD_ZjZ_ = MeanKLD(self.Q_PSPDF_Zj, self.Q_PSPDF_Z )  # I(zPSD;Zj|Z)
            I_zPSD_ZjFc_ =  MeanKLD(self.Q_PSPDF_ZjRptFC, self.P_PSPDF[None] ) # I(zPSD;Zj)
            I_zPSD_FaZj_ = MeanKLD(self.Q_PSPDF_ZjRptFCar, self.Q_PSPDF_ZjRptFC ) # I(zPSD;FC|Zj)
            I_fcPE_FcZj_ = MeanKLD(self.Q_PEPDF_ZjRptFC, self.Q_PEPDF_Batch) # I(fcPE;Zj)
            I_fcPE_FaZj_ = MeanKLD(self.Q_PEPDF_ZjRptFCar, self.Q_PEPDF_ZjRptFC) # I(fcPE;FC|Zj)


            print('I_zPSD_Z :', I_zPSD_Z_)
            self.SubResDic['I_zPSD_Z'].append(I_zPSD_Z_)
            self.I_zPSD_Z += I_zPSD_Z_

            print('I_zPSD_ZjZ :', I_zPSD_ZjZ_)
            self.SubResDic['I_zPSD_ZjZ'].append(I_zPSD_ZjZ_)
            self.I_zPSD_ZjZ += I_zPSD_ZjZ_

            print('I_zPSD_ZjFc :', I_zPSD_ZjFc_)
            self.SubResDic['I_zPSD_ZjFc'].append(I_zPSD_ZjFc_)
            self.I_zPSD_ZjFc += I_zPSD_ZjFc_

            print('I_zPSD_FaZj :', I_zPSD_FaZj_)
            self.SubResDic['I_zPSD_FaZj'].append(I_zPSD_FaZj_)
            self.I_zPSD_FaZj += I_zPSD_FaZj_

            print('I_fcPE_FcZj :', I_fcPE_FcZj_)
            self.SubResDic['I_fcPE_FcZj'].append(I_fcPE_FcZj_)
            self.I_fcPE_FcZj += I_fcPE_FcZj_

            print('I_fcPE_FaZj :', I_fcPE_FaZj_)
            self.SubResDic['I_fcPE_FaZj'].append(I_fcPE_FaZj_)
            self.I_fcPE_FaZj += I_fcPE_FaZj_
            
            
            ### --------------------------- Locating the candidate Z values that generate plausible signals ------------------------- ###
            self.H_zPSD_ZjFa = -np.sum(self.Q_PSPDF_ZjRptFCar * np.log(self.Q_PSPDF_ZjRptFCar), axis=-1)
            self.H_fcPE_ZjFa = np.mean(-np.sum(self.Q_PEPDF_ZjRptFCar * np.log(self.Q_PEPDF_ZjRptFCar), axis=-1), axis=-1)
            self.SumH_ZjFa = self.H_zPSD_ZjFa + self.H_fcPE_ZjFa

            # Calculating the mode-maximum frequency given the PSD from SigGen_ZjRptFCar.
            # Return shape: (Batch_size, N_sample, N_frequency)
            self.Q_PSPDF_ZjRptFCar_Local = FFT_PSD(self.SigGen_ZjRptFCar, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)

            # The 0 frequency is excluded as it represents the constant term; by adding 1 to the index, the frequency and index can be aligned to be the same.
            # Return shape: (Batch_size, N_sample)
            Max_Freq_Label = np.argmax(self.Q_PSPDF_ZjRptFCar_Local, axis=-1) + 1

            # Return shape: (Batch_size, )
            ModeMax_Freq = mode(Max_Freq_Label.T, axis=0, keepdims=False)[0]

            UniqSamp_Zj = self.Samp_ZjRPT.reshape(self.NMiniBat, self.NGen, -1)[:, 0]
            self.LocCandZs ( ModeMax_Freq, self.SumH_ZjFa, UniqSamp_Zj,)

            # Restructuring TrackerCandZ
            self.TrackerCandZ = {item[0]: {'TrackZLOC': np.concatenate(self.TrackerCandZ_Temp[item[0]]['TrackZLOC']), 
                                   'TrackZs': np.concatenate(self.TrackerCandZ_Temp[item[0]]['TrackZs']), 
                                   'TrackMetrics': np.concatenate(self.TrackerCandZ_Temp[item[0]]['TrackMetrics'])} 
                                     for item in self.TrackerCandZ_Temp.items() if len(item[1]['TrackZLOC']) > 0} 
            
            
        # Conducting the task iteration
        self.Iteration(TaskLogic)


        # CMI(V;Zj, Z)
        self.I_zPSD_Z /= (self.TotalIterSize)
        self.AggResDic['I_zPSD_Z'].append(self.I_zPSD_Z)
        self.I_zPSD_ZjZ /= (self.TotalIterSize)
        self.AggResDic['I_zPSD_ZjZ'].append(self.I_zPSD_ZjZ)
        self.CMI_zPSD_ZjZ = self.I_zPSD_Z + self.I_zPSD_ZjZ             
        self.AggResDic['CMI_zPSD_ZjZ'].append(self.CMI_zPSD_ZjZ)

        # CMI(V;FC,Zj)
        self.I_zPSD_ZjFc /= (self.TotalIterSize)
        self.AggResDic['I_zPSD_ZjFc'].append(self.I_zPSD_ZjFc)
        self.I_zPSD_FaZj /= (self.TotalIterSize)
        self.AggResDic['I_zPSD_FaZj'].append(self.I_zPSD_FaZj)
        self.CMI_zPSD_FcZj = self.I_zPSD_ZjFc + self.I_zPSD_FaZj       
        self.AggResDic['CMI_zPSD_FcZj'].append(self.CMI_zPSD_FcZj)

        # CMI(VE;FC,Zj)
        self.I_fcPE_FcZj /= (self.TotalIterSize)
        self.AggResDic['I_fcPE_FcZj'].append(self.I_fcPE_FcZj)
        self.I_fcPE_FaZj /= (self.TotalIterSize)
        self.AggResDic['I_fcPE_FaZj'].append(self.I_fcPE_FaZj)
        self.CMI_fcPE_FaFc = self.I_fcPE_FcZj + self.I_fcPE_FaZj    
        self.AggResDic['CMI_fcPE_FaFc'].append(self.CMI_fcPE_FaFc)

        
        
        
    
    
    ### -------------------------- Evaluating the performance of the model using only Z inputs  -------------------------- ###
    def Eval_Z (self, AnalData, SampModel, GenModel, Continue=True, SampZType='Model', SecDataType=None):

        ## Required parameters
        self.AnalData = AnalData             # The data to be used for analysis.
        self.SampModel = SampModel           # The model that samples Zs.
        self.GenModel = GenModel             # The model that generates signals based on given Zs and FCs.
        self.SecDataType = SecDataType       # The ancillary data-type: Use 'FCR' for FC values chosen randomly, 'FCA' for FC values given by arrange, 
                                             # and 'CON' for conditional inputs such as power spectral density.
        assert SecDataType in ['FCA','FCR','CON'], "Please verify the value of 'SecDataType'. Only 'FCA', 'FCR', or 'CON' are valid."
        

        ## Optional parameters with default values ##
        # Continue: Start from the beginning (Continue = False) vs. Continue where left off (Continue = True)
        self.SampZType = SampZType  # Z~ N(Zμ|y, σ) (SampZType = 'Model') vs. Z ~ N(0, ReparaStdZj) (SampZType = 'Random')


        ## Intermediate variables
        self.Ndata = len(AnalData) # The dimension size of the data.
        self.LatDim = SampModel.output.shape[-1] # The dimension size of Z.
        self.SigDim = AnalData.shape[-1] # The dimension (i.e., length) size of the raw signal.
        self.SubIterSize = self.Ndata//self.NMiniBat
        self.TotalIterSize = self.SubIterSize * self.SimSize


        # Functional trackers
        if Continue == False or not hasattr(self, 'iter'):
            self.sim, self.mini, self.iter = 0, 0, 0

            ## Result trackers
            self.SubResDic = {'I_zPSD_Z':[],'I_zPSD_ZjZ':[]}
            self.AggResDic = {'I_zPSD_Z':[],'I_zPSD_ZjZ':[],'CMI_zPSD_ZjZ':[]}
            self.BestZsMetrics = {i:[np.inf] for i in range(1, self.MaxFreq - self.MinFreq + 2)}
            self.TrackerCandZ_Temp = {i:{'TrackZLOC':[],'TrackZs':[],'TrackMetrics':[] } for i in range(1, self.MaxFreq - self.MinFreq + 2)} 
            self.I_zPSD_Z, self.I_zPSD_ZjZ = 0, 0



        ### ------------------------------------------------ Task logics ------------------------------------------------ ###

        # P(V=v)
        # Data shape: (N_frequency)
        self.P_PSPDF = FFT_PSD(self.AnalData, 'All', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)


        def TaskLogic(SubData):

            ### ------------------------------------------------ Sampling ------------------------------------------------ ###
            # Updating NMiniBat; If there is a remainder in Ndata/NMiniBat, NMiniBat must be updated." 
            self.NMiniBat = len(SubData)           

            # Sampling Samp_Z 
            self.Samp_Z = SamplingZ(SubData, self.SampModel, self.NMiniBat, self.NGen, 
                               BatchSize = self.SampBatchSize, GPU=self.GPU, SampZType=self.SampZType, ReparaStdZj=self.ReparaStdZj)

            # Selecting Samp_Zj from Samp_Z 
            ## For Samp_Zj, j is selected randomly across both the j and generation axes.
            self.Samp_Zj = SamplingZj (self.Samp_Z, self.NMiniBat, self.NGen, self.LatDim, self.NSelZ, Axis=1)



            ### ------------------------------------------------ Signal reconstruction ------------------------------------------------ ###
            ## Binding the samples together, generate signals through the model 
            Set_Zs = np.concatenate([self.Samp_Z, self.Samp_Zj])
            
            # Choosing GPU or CPU and generating signals 
            if self.GPU==False:
                with tf.device('/CPU:0'):
                    Set_Pred = self.GenModel.predict( Set_Zs, batch_size=self.GenBatchSize, verbose=1)
            else:
                Set_Pred = self.GenModel.predict( Set_Zs, batch_size=self.GenBatchSize, verbose=1)

            # Re-splitting predictions for each case
            Set_Pred = Set_Pred.reshape(-1, self.NMiniBat, self.NGen, self.SigDim)
            self.SigGen_Z, self.SigGen_Zj = [np.squeeze(SubPred) for SubPred in np.split(Set_Pred, 2) ]  



            ### ---------------------------- Cumulative Power Spectral Density (PSD) over each frequency -------------------------------- ###
            # Return shape: (Batch_size, N_frequency)
            self.Q_PSPDF_Z = FFT_PSD(self.SigGen_Z, 'Sample', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)
            self.Q_PSPDF_Zj = FFT_PSD(self.SigGen_Zj, 'Sample', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)




            ### ---------------------------------------- Conditional mutual information ---------------------------------------- ###
            # zPSD stands for z-wise power spectral density.
            I_zPSD_Z_ = MeanKLD(self.Q_PSPDF_Z, self.P_PSPDF[None] ) # I(zPSD;Z)
            I_zPSD_ZjZ_ = MeanKLD(self.Q_PSPDF_Zj, self.Q_PSPDF_Z )  # I(zPSD;Zj|Z)



            print('I_zPSD_Z :', I_zPSD_Z_)
            self.SubResDic['I_zPSD_Z'].append(I_zPSD_Z_)
            self.I_zPSD_Z += I_zPSD_Z_

            print('I_zPSD_ZjZ :', I_zPSD_ZjZ_)
            self.SubResDic['I_zPSD_ZjZ'].append(I_zPSD_ZjZ_)
            self.I_zPSD_ZjZ += I_zPSD_ZjZ_



            ### --------------------------- Locating the candidate Z values that generate plausible signals ------------------------- ###
            self.H_zPSD_Zj = -np.sum(self.Q_PSPDF_Zj * np.log(self.Q_PSPDF_Zj), axis=-1)

            # Calculating the mode-maximum frequency given the PSD from SigGen_ZjRptFCar.
            # Return shape: (Batch_size, N_sample, N_frequency)
            self.Q_PSPDF_Zj_Local = FFT_PSD(self.SigGen_Zj, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)

            # The 0 frequency is excluded as it represents the constant term; by adding 1 to the index, the frequency and index can be aligned to be the same.
            # Return shape: (Batch_size, N_sample)
            Max_Freq_Label = np.argmax(self.Q_PSPDF_Zj_Local, axis=-1) + 1

            # Return shape: (Batch_size, )
            ModeMax_Freq = mode(Max_Freq_Label.T, axis=0, keepdims=False)[0]
            self.LocCandZs ( ModeMax_Freq, self.H_zPSD_Zj, self.Samp_Zj)

            # Restructuring TrackerCandZ
            self.TrackerCandZ = {item[0]: {'TrackZLOC': np.concatenate(self.TrackerCandZ_Temp[item[0]]['TrackZLOC']), 
                                   'TrackZs': np.concatenate(self.TrackerCandZ_Temp[item[0]]['TrackZs']), 
                                   'TrackMetrics': np.concatenate(self.TrackerCandZ_Temp[item[0]]['TrackMetrics'])} 
                                     for item in self.TrackerCandZ_Temp.items() if len(item[1]['TrackZLOC']) > 0} 


        # Conducting the task iteration
        self.Iteration(TaskLogic)


        # CMI(V;Zj, Z)
        self.I_zPSD_Z /= (self.TotalIterSize)
        self.AggResDic['I_zPSD_Z'].append(self.I_zPSD_Z)
        self.I_zPSD_ZjZ /= (self.TotalIterSize)
        self.AggResDic['I_zPSD_ZjZ'].append(self.I_zPSD_ZjZ)
        self.CMI_zPSD_ZjZ = self.I_zPSD_Z + self.I_zPSD_ZjZ             
        self.AggResDic['CMI_zPSD_ZjZ'].append(self.CMI_zPSD_ZjZ)
        
        
        
        
        
        
        

    ### -------------------------- Evaluating the performance of the model using both Z and Conditions -------------------------- ###
    def Eval_Z_CON (self, AnalData, SampModel, GenModel, FcLimit=0.05,  WindowSize=3, Continue=True, SampZType='Model', SecDataType=None):
        
        ## Required parameters
        self.AnalData = AnalData             # The data to be used for analysis.
        self.SampModel = SampModel           # The model that samples Zs.
        self.GenModel = GenModel             # The model that generates signals based on given Zs and FCs.
        self.SecDataType = SecDataType       # The ancillary data-type: Use 'FCR' for FC values chosen randomly, 'FCA' for FC values given by arrange, 
                                             # and 'CON' for conditional inputs such as power spectral density.
        assert SecDataType in ['FCA','FCR','CON'], "Please verify the value of 'SecDataType'. Only 'FCA', 'FCR', or 'CON' are valid."
        
        
        ## Optional parameters with default values ##
        # WindowSize: The window size when calculating permutation entropy (default: 3)
        # Continue: Start from the beginning (Continue = False) vs. Continue where left off (Continue = True)
        self.SampZType = SampZType  # Z~ N(Zμ|y, σ) (SampZType = 'Model') vs. Z ~ N(0, ReparaStdZj) (SampZType = 'Random')
        self.FcLimit = FcLimit # The threshold value of the max of the FC value input into the generation model (default: 0.05, i.e., frequency 5 Hertz)      
            
        ## Intermediate variables
        self.Ndata = len(AnalData[0]) # The dimension size of the data.
        self.LatDim = SampModel.output.shape[-1] # The dimension size of Z.
        self.SigDim = AnalData[0].shape[-1] # The dimension (i.e., length) size of the raw signal.
        self.SubIterSize = self.Ndata//self.NMiniBat
        self.TotalIterSize = self.SubIterSize * self.SimSize
        
        
        # Functional trackers
        if Continue == False or not hasattr(self, 'iter'):
            self.sim, self.mini, self.iter = 0, 0, 0
        
            ## Result trackers
            self.SubResDic = {'I_zPSD_Z':[],'I_zPSD_ZjZ':[]}
            self.AggResDic = {'I_zPSD_Z':[],'I_zPSD_ZjZ':[],'CMI_zPSD_ZjZ':[]}
            self.BestZsMetrics = {i:[np.inf] for i in range(1, self.MaxFreq - self.MinFreq + 2)}
            self.TrackerCandZ_Temp = {i:{'TrackZLOC':[],'TrackZs':[],'TrackMetrics':[] } for i in range(1, self.MaxFreq - self.MinFreq + 2)} 
            self.I_zPSD_Z, self.I_zPSD_ZjZ = 0, 0
        
        


        ### ------------------------------------------------ Task logics ------------------------------------------------ ###
        
        # P(V=v)
        ## Data shape: (N_frequency)
        self.P_PSPDF = FFT_PSD(self.AnalData[0], 'All', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)
        
        
        def TaskLogic(SubData):
            
            

            ### ------------------------------------------------ Sampling ------------------------------------------------ ###
            # Updating NMiniBat; If there is a remainder in Ndata/NMiniBat, NMiniBat must be updated." 
            self.NMiniBat = len(SubData[0]) 

            # Sampling Samp_Z 
            self.Samp_Z = SamplingZ(SubData, self.SampModel, self.NMiniBat, self.NGen, 
                               BatchSize = self.SampBatchSize, GPU=self.GPU, SampZType=self.SampZType, SecDataType=self.SecDataType)

            # Selecting Samp_Zj from Samp_Z 
            ## For Samp_Zj, j is selected randomly across both the j and generation axes.
            self.Samp_Zj = SamplingZj (self.Samp_Z, self.NMiniBat, self.NGen, self.LatDim, self.NSelZ, Axis=1)


            # Setting CON 
            ## Shape of CON: (NMiniBat*NGen, CONDim) 
            self.CONRpt = np.repeat(SubData[1], NGen, axis=0)



            ### ------------------------------------------------ Signal reconstruction ------------------------------------------------ ###
            '''
            - To maximize the efficiency of GPU utilization, 
              we performed a binding operation on (NMiniBat, NGen, LatDim) for Zs and (NMiniBat, NGen, NFCs) for FCs, respectively, 
              transforming them to (NMiniBat * NGen, LatDim) and (NMiniBat * NGen, NFCs). 
              After the computation, we then reverted them back to their original dimensions.

            '''
            ## Binding the samples together, generate signals through the model 
            Set_Zs = np.concatenate([self.Samp_Z, self.Samp_Zj])
            Set_CONRpt = np.concatenate([self.CONRpt,  self.CONRpt])

            # Choosing GPU or CPU and generating signals 
            if self.GPU==False:
                with tf.device('/CPU:0'):
                    Set_Pred = self.GenModel.predict( [Set_Zs, Set_CONRpt], batch_size=self.GenBatchSize, verbose=1)
            else:
                Set_Pred = self.GenModel.predict( [Set_Zs, Set_CONRpt], batch_size=self.GenBatchSize, verbose=1)

            # Re-splitting predictions for each case
            Set_Pred = Set_Pred.reshape(-1, self.NMiniBat, self.NGen, self.SigDim)
            self.SigGen_Z, self.SigGen_Zj = [np.squeeze(SubPred) for SubPred in np.split(Set_Pred, 2)]  



            ### ---------------------------- Cumulative Power Spectral Density (PSD) over each frequency -------------------------------- ###
            # Return shape: (Batch_size, N_frequency)
            self.Q_PSPDF_Z = FFT_PSD(self.SigGen_Z, 'Sample', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)
            self.Q_PSPDF_Zj = FFT_PSD(self.SigGen_Zj, 'Sample', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)




            ### ---------------------------------------- Conditional mutual information ---------------------------------------- ###
            # zPSD stands for z-wise power spectral density.
            I_zPSD_Z_ = MeanKLD(self.Q_PSPDF_Z, self.P_PSPDF[None] ) # I(zPSD;Z)
            I_zPSD_ZjZ_ = MeanKLD(self.Q_PSPDF_Zj, self.Q_PSPDF_Z )  # I(zPSD;Zj|Z)



            print('I_zPSD_Z :', I_zPSD_Z_)
            self.SubResDic['I_zPSD_Z'].append(I_zPSD_Z_)
            self.I_zPSD_Z += I_zPSD_Z_

            print('I_zPSD_ZjZ :', I_zPSD_ZjZ_)
            self.SubResDic['I_zPSD_ZjZ'].append(I_zPSD_ZjZ_)
            self.I_zPSD_ZjZ += I_zPSD_ZjZ_



            ### --------------------------- Locating the candidate Z values that generate plausible signals ------------------------- ###
            self.H_zPSD_Zj = -np.sum(self.Q_PSPDF_Zj * np.log(self.Q_PSPDF_Zj), axis=-1)

            # Calculating the mode-maximum frequency given the PSD from SigGen_ZjRptFCar.
            # Return shape: (Batch_size, N_sample, N_frequency)
            self.Q_PSPDF_Zj_Local = FFT_PSD(self.SigGen_Zj, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)

            # The 0 frequency is excluded as it represents the constant term; by adding 1 to the index, the frequency and index can be aligned to be the same.
            # Return shape: (Batch_size, N_sample)
            Max_Freq_Label = np.argmax(self.Q_PSPDF_Zj_Local, axis=-1) + 1

            # Return shape: (Batch_size, )
            ModeMax_Freq = mode(Max_Freq_Label.T, axis=0, keepdims=False)[0]
            self.LocCandZs ( ModeMax_Freq, self.H_zPSD_Zj, self.Samp_Zj)

            # Restructuring TrackerCandZ
            self.TrackerCandZ = {item[0]: {'TrackZLOC': np.concatenate(self.TrackerCandZ_Temp[item[0]]['TrackZLOC']), 
                                   'TrackZs': np.concatenate(self.TrackerCandZ_Temp[item[0]]['TrackZs']), 
                                   'TrackMetrics': np.concatenate(self.TrackerCandZ_Temp[item[0]]['TrackMetrics'])} 
                                     for item in self.TrackerCandZ_Temp.items() if len(item[1]['TrackZLOC']) > 0} 


        # Conducting the task iteration
        self.Iteration(TaskLogic)


        # CMI(V;Zj, Z)
        self.I_zPSD_Z /= (self.TotalIterSize)
        self.AggResDic['I_zPSD_Z'].append(self.I_zPSD_Z)
        self.I_zPSD_ZjZ /= (self.TotalIterSize)
        self.AggResDic['I_zPSD_ZjZ'].append(self.I_zPSD_ZjZ)
        self.CMI_zPSD_ZjZ = self.I_zPSD_Z + self.I_zPSD_ZjZ             
        self.AggResDic['CMI_zPSD_ZjZ'].append(self.CMI_zPSD_ZjZ)

