import numpy as np
from scipy.stats import mode
import itertools
from tqdm import trange, tqdm

import tensorflow as tf
from tensorflow.keras import Model

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Utilities.AncillaryFunctions import FFT_PSD, ProbPermutation, MeanKLD, Sampler, SamplingZ, SamplingZj, GenConArange
from Utilities.Utilities import CompResource

       
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
        self.NSelZ = NSelZ                   # The size of js to be selected at the same time (default: 1).
        self.SampBatchSize = SampBatchSize   # The batch size during prediction of the sampling model.
        self.GenBatchSize= GenBatchSize      # The batch size during prediction of the generation model.
        self.GPU = GPU                       # GPU vs CPU during model predictions (i.e., for SampModel and GenModel).
        
        
        
    
    ''' ------------------------------------------------------ Ancillary Functions ------------------------------------------------------'''

    ### ----------- Searching for candidate Zj for plausible signal generation ----------- ###
    def LocCandZs (self, MaxFreq, EntH, Samp_Z, SecData=None):
        # Shape of MaxFreq: (NMiniBat*NGen, )
        # Shape of EntH: (NMiniBat*NGen, )
        # Shape of Samp_Z: (NMiniBat*NGen, LatDim)
        # Shape of SecData: (NMiniBat*NGen, SecDataDim)

        for Freq, _ in self.BestZsMetrics.items():
            FreqIdx = np.where(MaxFreq == Freq)[0]

            # Skipping the remainder of the code if there are no FreqIdx present at the predefined frequencies.
            if len(FreqIdx) <1: 
                continue;

            # Calculating the minimum of EntH (Entropy, H) and selecting candidate Z-values(CandZs)
            MinEntHIdx = np.argmin(EntH[FreqIdx]) 
            MinEntH = np.min(EntH[FreqIdx]) 
            CandZs = Samp_Z[[FreqIdx[MinEntHIdx]]]
            
            #tracking results
            self.TrackerCand_Temp[Freq]['TrackZs'].append(CandZs[None])
            self.TrackerCand_Temp[Freq]['TrackMetrics'].append(MinEntH[None])
            
            if SecData is not None: # for processing secondary data (SecData).
                CandSecData = SecData[[FreqIdx[MinEntHIdx]]]
                self.TrackerCand_Temp[Freq]['TrackSecData'].append(CandSecData[None])
            else:
                CandSecData = None

            # Updating the Min_SumH value if the current iteration value is smaller.
            if MinEntH < self.BestZsMetrics[Freq][0]:
                self.BestZsMetrics[Freq] = [MinEntH, CandZs, CandSecData]
                print('Candidate Z updated! ', 'Freq:', Freq, ', EntH:', np.round(MinEntH, 4))


    
    ### ------------------------  Selecting nested Z-LOC and Z values --------------------- ###
    def SubNestedZFix(self, SubTrackerCand):
                
        ''' Constructing the dictionary: {'KeyID': { 'TrackZs' : Zs, 'TrackSecData' : Secondary-data }}
           
           - Outer Dictionary:
             - Key (KeyID): A unique, sequentially increasing integer from 'Cnt'; That is, the sequence number in each frequency domain.
             - Value: An inner dictionary (explained below)
        
           - Inner Dictionary:
             - Key (TrackZs) : Value (Tracked Z-value matrix)
             - Key (TrackSecData) : Values (Tracked secondary data matrix)
        '''
        
        Cnt = itertools.count()
        if self.SecDataType == False:
            Results = {next(Cnt):{ 'TrackZs' : TrackZs} 
                        for TrackZs, TrackMetrics 
                        in zip(SubTrackerCand['TrackZs'], SubTrackerCand['TrackMetrics'])
                        if TrackMetrics < self.MetricCut }

        else:
            Results = {next(Cnt):{ 'TrackZs' : TrackZs, 'TrackSecData' : TrackSecData} 
                        for TrackZs, TrackSecData, TrackMetrics 
                        in zip(SubTrackerCand['TrackZs'], SubTrackerCand['TrackSecData'], SubTrackerCand['TrackMetrics'])
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
                if self.SecDataType == 'CONR' or self.SecDataType == 'CONA' : 
                    SplitData = [np.array_split(sub, self.SubIterSize) for sub in self.AnalData]   

                else: # For models with a single input such as VAE and TCVAE.
                    SplitData = np.array_split(self.AnalData, self.SubIterSize)    

                for mini in range(self.mini, self.SubIterSize):
                    self.mini = mini
                    self.iter += 1
                    print()

                    # Core part; the task logic as the function
                    if self.SecDataType  == 'CONR' or self.SecDataType == 'CONA' : 
                        TaskLogic([subs[mini] for subs in SplitData])
                    else:
                        TaskLogic(SplitData[mini])

                    t.update(1)
    
    
    
    ### ------------------- Selecting post-sampled Z values for generating plausible signals ------------------- ###
    def SelPostSamp (self, MetricCut=np.inf, BestZsMetrics=None, TrackerCand=None, SavePath=None ):
        
        ## Optional parameters
        # MetricCut: The threshold value for selecting Zs whose Entropy of PSD (i.e., SumH) is less than the MetricCut
        self.MetricCut = MetricCut

        # Setting arguments
        BestZsMetrics = self.BestZsMetrics if BestZsMetrics is None else BestZsMetrics
        TrackerCand = self.TrackerCand if TrackerCand is None else TrackerCand
                
        # Exploring FreqIDs available for signal generation  
        ## FreqIDs such as [9, 10, 11 ..... 45]
        ## item[0] contains frequency domains
        ## item[1][0] contains metrics
        self.CandFreqIDs = [item[0] for item in BestZsMetrics.items() if item[1][0] != np.inf ]
            
        
        # Selecting nested Z-LOC and Z values
        '''  Constructing the dictionary : {'FreqID' : {'SubKeys' : { 'TrackZs' : Zs, 'TrackSecData' : Secondary-data }}}
        
           - Outermost Dictionary:
             - Key (FreqID): Represents frequency identifiers.
             - Value: A second-level dictionary (explained below).
        
           - Second-level Dictionary:
             - Key (SubKeys): Represents some sub-category or sub-key for the given 'FreqID'.
             - Value: A third-level dictionary (explained below).
        
           - Third-level Dictionary:
             - Key (TrackZs) : Value (Tracked Z-value matrix)
             - Key (TrackSecData) : Values (Tracked secondary data matrix)
             
        '''
        self.PostSamp = {FreqID : self.SubNestedZFix(TrackerCand[FreqID], ) for FreqID in self.CandFreqIDs}

        
        
        # Counting the number of obs in NestedZs
        NPostZs =0 
        for item in self.PostSamp.items():
            NPostZs += len(item[1])

        print('The total number of sets in NestedZs:', NPostZs)

        
        return self.PostSamp
    
    
        
    ### -------------- Evaluating the KLD between the PSD of the true signals and the generated signals ---------------- ###
    def KLD_TrueGen (self, PostSamp=None, AnalData=None, SecDataType=None, FcLimit=None, RepeatSize=1, PlotDist=True):
    
        ## Required parameters
        # PostSamp: The post-sampled data for generating signals with the shape of ({'FreqID': {'SubKeys': {'TrackZs': Zs, 'TrackSecData': Secondary-data}}}).
        # AnalData: The raw true signals for obtaining the population PSD.    
        
        ## Optional parameters
        # RepeatSize: The number of iterations to repetitively generate identical PostSampZ; 
                    # this is to observe variations in other inputs such as FCs while PostSampZ remains constant.
        # SecDataType: Secondary data type; Use 'FCR' or 'FCA' for random FC or arranged FC values, respectively,
                      # 'CONA' or 'CONR' for random conditional inputs or arranged conditional inputs, respectively.
                      # 'False' for models without secondary-data inputs.
        # FcLimit: The threshold value of the max of the FC value input into the generation model (default: 0.05, i.e., frequency 5 Hertz).
        # PostSamp: The selected sampled data.

        
        # Setting arguments
        PostSamp = self.PostSamp if PostSamp is None else PostSamp
        FcLimit = self.FcLimit if FcLimit is None and hasattr(self, 'FcLimit') else FcLimit
        AnalData = self.AnalData if AnalData is None else AnalData
        SecDataType = self.SecDataType if SecDataType is None else SecDataType
        
        # Converting the dictionary to the list type.
        PostZsList = []
        PostSecDataList = []

        for Freq, Subkeys in PostSamp.items():
            for Subkeys, Values in Subkeys.items():
                PostZsList.append(np.array(Values['TrackZs']))
                if 'TrackSecData' in Values.keys(): 
                    PostSecDataList.append(np.array(Values['TrackSecData']))
        
        # Converting the list type to the np-data type.
        PostZsList = np.concatenate(PostZsList)
        if SecDataType is not False:  # it means there are secondary-data inputs
            PostSecDataList = np.concatenate(PostSecDataList)
        
        
        # Data binding for the model input
        if SecDataType == 'FCA' : 
            Data = [PostSecDataList[:, :2], PostSecDataList[:, 2:], PostZsList]
        
        elif SecDataType == 'FCR':
            PostZsList = np.repeat(PostZsList, RepeatSize, axis=0)
            PostSecDataList = np.repeat(PostSecDataList, RepeatSize, axis=0)
            PostSecDataList = np.random.permutation(np.random.permutation(PostSecDataList.T).T)
            Data = [PostSecDataList[:, :2], PostSecDataList[:, 2:], PostZsList]
        
        elif SecDataType == 'CONA': 
            Data = [PostZsList, PostSecDataList]
        
        elif SecDataType == 'CONR':  
            PostZsList = np.repeat(PostZsList, RepeatSize, axis=0)
            PostSecDataList = np.repeat(PostSecDataList, RepeatSize, axis=0)
            PostSecDataList = np.random.permutation(PostSecDataList.T).T
            Data = [PostZsList, PostSecDataList]
            
        elif SecDataType == False :
            Data = PostZsList
            
          
        # Generating signals
        self.GenSamp = CompResource (self.GenModel, Data, BatchSize=self.GenBatchSize, GPU=self.GPU)
            

        # Calculating the KLD between the PSD of the true signals and the generated signals    
        PSDGenSamp =  FFT_PSD(self.GenSamp, 'All', MinFreq = 1, MaxFreq = 51)
        if SecDataType == 'CONA' or SecDataType == 'CONR'  : # Conditional inputs such as power spectral density
            PSDTrueData =  FFT_PSD(AnalData[0], 'All', MinFreq = 1, MaxFreq = 51)
        else:
            PSDTrueData =  FFT_PSD(AnalData, 'All', MinFreq = 1, MaxFreq = 51)
            
        self.KldPSD_GenTrue = MeanKLD(PSDGenSamp, PSDTrueData)
        self.KldPSD_TrueGen  = MeanKLD(PSDTrueData, PSDGenSamp)
        self.MeanKld_GTTG = (self.KldPSD_GenTrue + self.KldPSD_TrueGen) / 2
        

        print('KldPSD_GenTrue: ', self.KldPSD_GenTrue)
        print('KldPSD_TrueGen: ', self.KldPSD_TrueGen)
        print('MeanKld_GTTG: ', self.MeanKld_GTTG)

        if PlotDist==True:
            plt.plot(PSDGenSamp, c='green', label='Generated')
            plt.plot(PSDTrueData, c='orange', label='True')
            plt.fill_between(np.arange(len(PSDTrueData)), PSDTrueData, color='orange', alpha=0.5)
            plt.fill_between(np.arange(len(PSDGenSamp)), PSDGenSamp, color='green', alpha=0.5)
            plt.legend()    
    
        
    
    
    
    ''' ------------------------------------------------------ Main Functions ------------------------------------------------------'''
    
    ### -------------------------- Evaluating the performance of the model using both Z and FC inputs  -------------------------- ###
    def Eval_ZFC (self, AnalData, SampModel, GenModel, FC_ArangeInp, FcLimit=0.05,  WindowSize=3, Continue=True, SampZType='ModelRptA',  SecDataType='FCA'):
        
        ## Required parameters
        self.AnalData = AnalData             # The data to be used for analysis.
        self.SampModel = SampModel           # The model that samples Zs.
        self.GenModel = GenModel             # The model that generates signals based on given Zs and FCs.
        self.FC_ArangeInp = FC_ArangeInp     # The 2D matrix (N_sample, NFCs) containing FCs values that the user creates and inputs directly.
        
        assert SecDataType in ['FCA','FCR','CONR','CONA', False], "Please verify the value of 'SecDataType'. Only 'FCA', 'FCR', 'CONR', 'CONA'  or False are valid."
        
        
        ## Optional parameters with default values ##
        # WindowSize: The window size when calculating the permutation sets (default: 3)
        # Continue: Start from the beginning (Continue = False) vs. Continue where left off (Continue = True)
        self.SampZType = SampZType           # Z~ N(Zμ|y, σ) (SampZType = 'Model') vs. Z ~ N(0, ReparaStdZj) (SampZType = 'Random')
        self.FcLimit = FcLimit               # The threshold value of the max of the FC value input into the generation model (default: 0.05, i.e., frequency 5 Hertz)      
        self.SecDataType = SecDataType       # The ancillary data-type: Use 'FCR' for FC values chosen randomly, 'FCA' for FC values given by arrange, 
                                             # and 'CON' for conditional inputs such as power spectral density.
        
        
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
            self.SubResDic = {'I_zPSD_Z':[],'I_zPSD_ZjZ':[],'I_zPSD_ZjFc':[],'I_zPSD_FaZj':[],'I_fcPE_ZjFc':[],'I_fcPE_FaZj':[]}
            self.AggResDic = {'I_zPSD_Z':[],'I_zPSD_ZjZ':[],'I_zPSD_ZjFc':[],'I_zPSD_FaZj':[],'I_fcPE_ZjFc':[],'I_fcPE_FaZj':[], 
                         'MI_zPSD_ZjZ':[], 'MI_zPSD_FcZj':[], 'MI_fcPE_FaFc':[]}
            self.BestZsMetrics = {i:[np.inf] for i in range(1, self.MaxFreq - self.MinFreq + 2)}
            self.TrackerCand_Temp = {i:{'TrackSecData':[],'TrackZs':[],'TrackMetrics':[] } for i in range(1, self.MaxFreq - self.MinFreq + 2)} 
            self.I_zPSD_Z, self.I_zPSD_ZjZ, self.I_zPSD_ZjFc, self.I_zPSD_FaZj, self.I_fcPE_ZjFc, self.I_fcPE_FaZj = 0,0,0,0,0,0
        
        
        
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
            self.Samp_Zj = SamplingZj (self.Samp_Z, self.NMiniBat, self.NGen, self.LatDim, self.NSelZ, ZjType='AllRand')
            ## For Samp_ZjRPT, the same j is selected in all generations within a mini-batch.
            self.Samp_ZjRPT = SamplingZj (self.Samp_Z, self.NMiniBat, self.NGen, self.LatDim, self.NSelZ, ZjType='RptBat')



            # Sampling FCs
            ## Shape of FCs: (NMiniBat*NGen, NFCs) instead of (NMiniBat, NGen, NFCs) for optimal use of GPU
            FCs = np.random.rand(self.NMiniBat,  self.NGen, self.NFCs) * FcLimit

            # Generating FC values sorted in ascending order at the NGen index.
            self.FC_Arange = np.sort(FCs, axis=1).reshape(self.NMiniBat*self.NGen, self.NFCs)
            self.FCs = np.reshape(FCs, (self.NMiniBat* self.NGen, self.NFCs))




            ### ------------------------------------------------ Signal reconstruction ------------------------------------------------ ###
            '''
            - To maximize the efficiency of GPU utilization, 
              we performed a binding operation on (NMiniBat, NGen, LatDim) for Zs and (NMiniBat, NGen, NFCs) for FCs, respectively, 
              transforming them to (NMiniBat * NGen, LatDim) and (NMiniBat * NGen, NFCs). 
              After the computation, we then reverted them back to their original dimensions.

            '''
            ## Binding the samples together, generate signals through the model 
            Set_FCs = np.concatenate([self.FCs,   self.FCs,     self.FCs,         self.FC_Arange]) 
            Set_Zs = np.concatenate([self.Samp_Z,  self.Samp_ZjRPT,  self.Samp_ZjRPT,  self.Samp_ZjRPT])
            '''Mathematically, Samp_ZjRPT seems more valid than Samp_Zj on the 2nd element in Set_Zs, thus a test needs to be conducted. 
             Expect cascading effects when modifying this; ensure to check variable names carefully while making changes.'''
            Data = [Set_FCs[:, :2], Set_FCs[:, 2:], Set_Zs]


            # Choosing GPU or CPU and generating signals
            Set_Pred = CompResource (self.GenModel, Data, BatchSize=self.GenBatchSize, GPU=self.GPU)


            # Re-splitting predictions for each case
            Set_Pred = Set_Pred.reshape(-1, self.NMiniBat, self.NGen, self.SigDim)
            self.SigGen_Z, self.SigGen_Zj, self.SigGen_ZjRptFC, self.SigGen_ZjRptFCar = [np.squeeze(SubPred) for SubPred in np.split(Set_Pred, 4)]  



            ### ---------------------------- Cumulative Power Spectral Density (PSD) over each frequency -------------------------------- ###
            # Return shape: (Batch_size, N_frequency)
            self.Q_PSPDF_Z = FFT_PSD(self.SigGen_Z, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).mean(axis=1)
            self.Q_PSPDF_Zj = FFT_PSD(self.SigGen_Zj, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).mean(axis=1)
            self.Q_PSPDF_ZjRptFC = FFT_PSD(self.SigGen_ZjRptFC, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).mean(axis=1)
            self.Q_PSPDF_ZjRptFCar = FFT_PSD(self.SigGen_ZjRptFCar, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).mean(axis=1)

            # Return shape: (Batch_size, N_frequency, N_sample)
            self.SubPSPDF_ZjRptFC = FFT_PSD(self.SigGen_ZjRptFC, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).transpose(0,2,1)
            self.SubPSPDF_ZjRptFCar = FFT_PSD(self.SigGen_ZjRptFCar, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).transpose(0,2,1)

            # Return shape: (Batch_size, 1, N_frequency)
            self.SubPSPDF_Batch = FFT_PSD(SubData[:,None], 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)


            ### ---------------------------- Permutation density given PSD over each generation -------------------------------- ###
            # Return shape: (Batch_size, N_frequency, N_permutation_cases)
            self.Q_PDPSD_ZjRptFC = ProbPermutation(self.SubPSPDF_ZjRptFC, WindowSize=WindowSize)
            self.Q_PDPSD_ZjRptFCar = ProbPermutation(self.SubPSPDF_ZjRptFCar, WindowSize=WindowSize)
            self.Q_PDPSD_Batch = ProbPermutation(self.SubPSPDF_Batch, WindowSize=WindowSize)


            ### ---------------------------------------- Mutual information ---------------------------------------- ###
            # zPSD and fcPE stand for z-wise power spectral density and fc-wise permutation sets, respectively.
            I_zPSD_Z_ = MeanKLD(self.Q_PSPDF_Z, self.P_PSPDF[None] ) # I(zPSD;Z)
            I_zPSD_ZjZ_ = MeanKLD(self.Q_PSPDF_Zj, self.Q_PSPDF_Z )  # I(zPSD;Zj|Z)
            I_zPSD_ZjFc_ =  MeanKLD(self.Q_PSPDF_ZjRptFC, self.P_PSPDF[None] ) # I(zPSD;Zj)
            I_zPSD_FaZj_ = MeanKLD(self.Q_PSPDF_ZjRptFCar, self.Q_PSPDF_ZjRptFC ) # I(zPSD;FC|Zj)
            I_fcPE_ZjFc_ = MeanKLD(self.Q_PDPSD_ZjRptFC, self.Q_PDPSD_Batch) # I(fcPE;FC,Zj)
            I_fcPE_FaZj_ = MeanKLD(self.Q_PDPSD_ZjRptFCar, self.Q_PDPSD_ZjRptFC) # I(fcPE;FCa,Zj)


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

            print('I_fcPE_ZjFc :', I_fcPE_ZjFc_)
            self.SubResDic['I_fcPE_ZjFc'].append(I_fcPE_ZjFc_)
            self.I_fcPE_ZjFc += I_fcPE_ZjFc_

            print('I_fcPE_FaZj :', I_fcPE_FaZj_)
            self.SubResDic['I_fcPE_FaZj'].append(I_fcPE_FaZj_)
            self.I_fcPE_FaZj += I_fcPE_FaZj_
            
            
            ### --------------------------- Locating the candidate Z values that generate plausible signals ------------------------- ###
            # Calculating the entropies given the probability density function of the power spectral.
            ## This indicates which frequency is most activated in the generated signal.
            EntH = -np.sum(self.SubPSPDF_ZjRptFCar * np.log(self.SubPSPDF_ZjRptFCar), axis=1).ravel()

            # Getting the maximum frequency given the PSD from SubPSPDF_ZjRptFCar.
            ## The 0 frequency is excluded as it represents the constant term; by adding 1 to the index, the frequency and index can be aligned to be the same.
            ## Return shape: (Batch_size, N_sample) -> (Batch_size x N_sample) for the computational efficiency (i.e, ravel function applied)
            MaxFreq = np.argmax(self.SubPSPDF_ZjRptFCar, axis=1).ravel() + 1
            
            self.LocCandZs ( MaxFreq, EntH, self.Samp_ZjRPT,  self.FC_Arange)

            # Restructuring TrackerCand
            ## item[0] contains frequency domains
            ## item[1] contains tracked Z values, 2nd data, and metrics
            self.TrackerCand = {item[0]: {'TrackZs': np.concatenate(self.TrackerCand_Temp[item[0]]['TrackZs']), 
                                          'TrackSecData': np.concatenate(self.TrackerCand_Temp[item[0]]['TrackSecData']), 
                                          'TrackMetrics': np.concatenate(self.TrackerCand_Temp[item[0]]['TrackMetrics'])} 
                                          for item in self.TrackerCand_Temp.items() if len(item[1]['TrackSecData']) > 0} 
            
            
        # Conducting the task iteration
        self.Iteration(TaskLogic)


        # MI(V;Zj,Z)
        self.I_zPSD_Z /= (self.TotalIterSize)
        self.AggResDic['I_zPSD_Z'].append(self.I_zPSD_Z)
        self.I_zPSD_ZjZ /= (self.TotalIterSize)
        self.AggResDic['I_zPSD_ZjZ'].append(self.I_zPSD_ZjZ)
        self.MI_zPSD_ZjZ = self.I_zPSD_Z + self.I_zPSD_ZjZ             
        self.AggResDic['MI_zPSD_ZjZ'].append(self.MI_zPSD_ZjZ)

        # MI(V;FC,Zj)
        self.I_zPSD_ZjFc /= (self.TotalIterSize)
        self.AggResDic['I_zPSD_ZjFc'].append(self.I_zPSD_ZjFc)
        self.I_zPSD_FaZj /= (self.TotalIterSize)
        self.AggResDic['I_zPSD_FaZj'].append(self.I_zPSD_FaZj)
        self.MI_zPSD_FcZj = self.I_zPSD_ZjFc + self.I_zPSD_FaZj       
        self.AggResDic['MI_zPSD_FcZj'].append(self.MI_zPSD_FcZj)

        # I(VE;FCa,Zj) - I(VE;FCr,Zj)
        self.I_fcPE_ZjFc /= (self.TotalIterSize)
        self.AggResDic['I_fcPE_ZjFc'].append(self.I_fcPE_ZjFc)
        self.I_fcPE_FaZj /= (self.TotalIterSize)
        self.AggResDic['I_fcPE_FaZj'].append(self.I_fcPE_FaZj)
        self.MI_fcPE_FaFc = self.I_fcPE_FaZj - self.I_fcPE_ZjFc
        self.AggResDic['MI_fcPE_FaFc'].append(self.MI_fcPE_FaFc)

        
        
        
    
    
    ### -------------------------- Evaluating the performance of the model using only Z inputs  -------------------------- ###
    def Eval_Z (self, AnalData, SampModel, GenModel, Continue=True, SampZType='ModelRptA',  SecDataType=False):

        ## Required parameters
        self.AnalData = AnalData             # The data to be used for analysis.
        self.SampModel = SampModel           # The model that samples Zs.
        self.GenModel = GenModel             # The model that generates signals based on given Zs and FCs.
        self.SecDataType = SecDataType       # The ancillary data-type: Use 'FCR' for FC values chosen randomly, 'FCA' for FC values given by arrange, 
                                             # and 'CON' for conditional inputs such as power spectral density.
        assert SecDataType in ['FCA','FCR','CONR','CONA', False], "Please verify the value of 'SecDataType'. Only 'FCA', 'FCR', 'CONR', 'CONA'  or False are valid."
        

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
            self.AggResDic = {'I_zPSD_Z':[],'I_zPSD_ZjZ':[],'MI_zPSD_ZjZ':[]}
            self.BestZsMetrics = {i:[np.inf] for i in range(1, self.MaxFreq - self.MinFreq + 2)}
            self.TrackerCand_Temp = {i:{'TrackZs':[],'TrackMetrics':[] } for i in range(1, self.MaxFreq - self.MinFreq + 2)} 
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
            self.Samp_Zj = SamplingZj (self.Samp_Z, self.NMiniBat, self.NGen, self.LatDim, self.NSelZ, ZjType='AllRand')



            ### ------------------------------------------------ Signal reconstruction ------------------------------------------------ ###
            ## Binding the samples together, generate signals through the model 
            Set_Zs = np.concatenate([self.Samp_Z, self.Samp_Zj])
            
            # Choosing GPU or CPU and generating signals
            Set_Pred  = CompResource (self.GenModel, Set_Zs, BatchSize=self.GenBatchSize, GPU=self.GPU)

            # Re-splitting predictions for each case
            Set_Pred = Set_Pred.reshape(-1, self.NMiniBat, self.NGen, self.SigDim)
            self.SigGen_Z, self.SigGen_Zj = [np.squeeze(SubPred) for SubPred in np.split(Set_Pred, 2) ]  



            ### ---------------------------- Cumulative Power Spectral Density (PSD) over each frequency -------------------------------- ###
            # Return shape: (Batch_size, N_frequency)
            self.Q_PSPDF_Z = FFT_PSD(self.SigGen_Z, 'Sample', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)
            self.Q_PSPDF_Zj = FFT_PSD(self.SigGen_Zj, 'Sample', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)




            ### ---------------------------------------- Mutual information ---------------------------------------- ###
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
            # Return shape: (Batch_size, N_sample, N_frequency)
            self.SubPSPDF_Zj = FFT_PSD(self.SigGen_Zj, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)
            
            # Calculating the entropies given the probability density function of the power spectral.
            ## This indicates which frequency is most activated in the generated signal.
            EntH = -np.sum(self.SubPSPDF_Zj * np.log(self.SubPSPDF_Zj), axis=1).ravel()

            # Getting the maximum frequency given the PSD from SigGen_Zj.
            # The 0 frequency is excluded as it represents the constant term; by adding 1 to the index, the frequency and index can be aligned to be the same.
            # Return shape: (Batch_size, N_sample)
            MaxFreq = np.argmax(self.SubPSPDF_Zj, axis=1).ravel() + 1

            self.LocCandZs ( MaxFreq, EntH, self.Samp_Zj)

            # Restructuring TrackerCand
            ## item[0] contains frequency domains
            ## item[1] contains tracked Z values and metrics
            self.TrackerCand = {item[0]: {'TrackZs': np.concatenate(self.TrackerCand_Temp[item[0]]['TrackZs']), 
                                   'TrackMetrics': np.concatenate(self.TrackerCand_Temp[item[0]]['TrackMetrics'])} 
                                     for item in self.TrackerCand_Temp.items() if len(item[1]['TrackZs']) > 0} 


        # Conducting the task iteration
        self.Iteration(TaskLogic)


        # MI(V;Zj,Z)
        self.I_zPSD_Z /= (self.TotalIterSize)
        self.AggResDic['I_zPSD_Z'].append(self.I_zPSD_Z)
        self.I_zPSD_ZjZ /= (self.TotalIterSize)
        self.AggResDic['I_zPSD_ZjZ'].append(self.I_zPSD_ZjZ)
        self.MI_zPSD_ZjZ = self.I_zPSD_Z + self.I_zPSD_ZjZ             
        self.AggResDic['MI_zPSD_ZjZ'].append(self.MI_zPSD_ZjZ)
        
        
        
        
        
        
        

    ### -------------------------- Evaluating the performance of the model using both Z and Conditions -------------------------- ###
    def Eval_ZCON (self, AnalData, SampModel, GenModel,  WindowSize=3, SampZType='ModelRptA', SecDataType=None, Continue=True):
        
        ## Required parameters
        self.AnalData = AnalData             # The data to be used for analysis.
        self.SampModel = SampModel           # The model that samples Zs.
        self.GenModel = GenModel             # The model that generates signals based on given Zs and FCs.
        
        assert SecDataType in ['FCA','FCR','CONR','CONA', False], "Please verify the value of 'SecDataType'. Only 'FCA', 'FCR', 'CONR', 'CONA'  or False are valid."
        
        
        
        ## Optional parameters with default values ##
        # WindowSize: The window size when calculating the permutation sets (default: 3).
        # Continue: Start from the beginning (Continue = False) vs. Continue where left off (Continue = True).
        self.SampZType = SampZType  # Z~ N(Zμ|y, σ) (SampZType = 'Model') vs. Z ~ N(0, ReparaStdZj) (SampZType = 'Random').
        self.SecDataType = SecDataType       # The ancillary data-type: Use 'FCR' for FC values chosen randomly, 'FCA' for FC values given by arrange, 
                                             # and 'CON' for conditional inputs such as power spectral density.
        
        

        ## Intermediate variables
        self.Ndata = len(AnalData[0]) # The dimension size of the data.
        self.LatDim = SampModel.output.shape[-1] # The dimension size of Z.
        self.SigDim = AnalData[0].shape[-1] # The dimension (i.e., length) size of the raw signal.
        self.CondDim = AnalData[1].shape[-1] # The dimension size of the conditional inputs.
        self.SubIterSize = self.Ndata//self.NMiniBat
        self.TotalIterSize = self.SubIterSize * self.SimSize
        
        assert self.NGen >= self.CondDim, "NGen must be greater than or equal to CondDim for the evaluation."

        
        # Functional trackers
        if Continue == False or not hasattr(self, 'iter'):
            self.sim, self.mini, self.iter = 0, 0, 0
        
            ## Result trackers
            self.SubResDic = {'I_zPSD_Z':[],'I_zPSD_ZjZ':[],'I_zPSD_ZjCr':[],'I_zPSD_CaZj':[],'I_fcPE_ZjCr':[],'I_fcPE_CaZj':[]}
            self.AggResDic = {'I_zPSD_Z':[],'I_zPSD_ZjZ':[],'I_zPSD_ZjCr':[],'I_zPSD_CaZj':[],'I_fcPE_ZjCr':[],'I_fcPE_CaZj':[], 
                              'MI_zPSD_ZjZ':[], 'MI_zPSD_CrZj':[], 'MI_fcPE_CaCr':[]}
            self.BestZsMetrics = {i:[np.inf] for i in range(1, self.MaxFreq - self.MinFreq + 2)}
            self.TrackerCand_Temp = {i:{'TrackSecData':[],'TrackZs':[],'TrackMetrics':[] } for i in range(1, self.MaxFreq - self.MinFreq + 2)} 
            self.I_zPSD_Z, self.I_zPSD_ZjZ, self.I_zPSD_ZjCr, self.I_zPSD_CaZj, self.I_fcPE_ZjCr, self.I_fcPE_CaZj = 0,0,0,0,0,0
        
        


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
            self.Samp_Zj = SamplingZj (self.Samp_Z, self.NMiniBat, self.NGen, self.LatDim, self.NSelZ, ZjType='AllRand')
            
            ## For Samp_ZjRPT, the same j is selected NGen times in all generations within a mini-batch.
            self.Samp_ZjRPT = SamplingZj (self.Samp_Z, self.NMiniBat, self.NGen, self.LatDim, self.NSelZ, ZjType='RptBat')
  
            
            # Processing Conditional information 
            ## Generating CON_Arange
            CON_Arange = GenConArange(self.AnalData[1], self.NGen)
            self.CON_Arange = np.tile(CON_Arange[None], (self.NMiniBat, 1,1)).reshape(self.NMiniBat*self.NGen, -1)
                        
            ## Generating CONRand
            #self.CONRand = np.random.rand(self.NMiniBat * self.NGen, self.CondDim)
            CONRand = np.random.permutation(self.AnalData[1])[np.random.choice(self.Ndata, self.NMiniBat*self.NGen)]
            self.CONRand= np.random.permutation(CONRand.T).T
            
            
            
            ### ------------------------------------------------ Signal reconstruction ------------------------------------------------ ###
            '''
            - To maximize the efficiency of GPU utilization, 
              we performed a binding operation on (NMiniBat, NGen, LatDim) for Zs and (NMiniBat, NGen, NFCs) for FCs, respectively, 
              transforming them to (NMiniBat * NGen, LatDim) and (NMiniBat * NGen, NFCs). 
              After the computation, we then reverted them back to their original dimensions.
            '''
            
            ## Binding the samples together, generate signals through the model 
            Set_CONs = np.concatenate([self.CONRand,  self.CONRand, self.CONRand, self.CON_Arange])
            Set_Zs = np.concatenate([self.Samp_Z, self.Samp_Zj, self.Samp_ZjRPT, self.Samp_ZjRPT ])
            Data = [Set_Zs, Set_CONs]
            
            
            # Choosing GPU or CPU and generating signals
            Set_Pred  = CompResource (self.GenModel, Data, BatchSize=self.GenBatchSize, GPU=self.GPU)

            # Re-splitting predictions for each case
            Set_Pred = Set_Pred.reshape(-1, self.NMiniBat, self.NGen, self.SigDim)
            # Shape of each generation: NMiniBat, NGen, SigDim
            self.SigGen_Z, self.SigGen_Zj, self.SigGen_ZjRptCONr, self.SigGen_ZjRptCONa = [np.squeeze(SubPred) for SubPred in np.split(Set_Pred, 4)]  



            ### ---------------------------- Cumulative Power Spectral Density (PSD) over each frequency -------------------------------- ###
            # Return shape: (Batch_size, N_frequency)
            self.Q_PSPDF_Z = FFT_PSD(self.SigGen_Z, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).mean(axis=1)
            self.Q_PSPDF_Zj = FFT_PSD(self.SigGen_Zj, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).mean(axis=1)
            self.Q_PSPDF_ZjRptCONr = FFT_PSD(self.SigGen_ZjRptCONr, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).mean(axis=1)
            self.Q_PSPDF_ZjRptCONa = FFT_PSD(self.SigGen_ZjRptCONa, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).mean(axis=1)

            # Return shape: (Batch_size, N_frequency, N_sample)
            self.SubPSPDF_ZjRptCONr = FFT_PSD(self.SigGen_ZjRptCONr, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).transpose(0,2,1)
            self.SubPSPDF_ZjRptCONa = FFT_PSD(self.SigGen_ZjRptCONa, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).transpose(0,2,1)

            # Return shape: (Batch_size, 1, N_frequency)
            self.SubPSPDF_Batch = FFT_PSD(SubData[0][:,None], 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq) #


            
            ### ---------------------------- Permutation density given PSD over each generation -------------------------------- ###
            # Return shape: (Batch_size, N_frequency, N_permutation_cases)
            self.Q_PDPSD_ZjRptCONr = ProbPermutation(self.SubPSPDF_ZjRptCONr, WindowSize=WindowSize)
            self.Q_PDPSD_ZjRptCONa = ProbPermutation(self.SubPSPDF_ZjRptCONa, WindowSize=WindowSize)
            self.Q_PDPSD_Batch = ProbPermutation(self.SubPSPDF_Batch, WindowSize=WindowSize)
            
            
            
            ### ---------------------------------------- Mutual information ---------------------------------------- ###
            # zPSD stands for z-wise power spectral density.
            I_zPSD_Z_ = MeanKLD(self.Q_PSPDF_Z, self.P_PSPDF[None] ) # I(zPSD;Z)
            I_zPSD_ZjZ_ = MeanKLD(self.Q_PSPDF_Zj, self.Q_PSPDF_Z )  # I(zPSD;Zj|Z)
            I_zPSD_ZjCr_ =  MeanKLD(self.Q_PSPDF_ZjRptCONr, self.P_PSPDF[None] ) # I(zPSD;Zj)
            I_zPSD_CaZj_ = MeanKLD(self.Q_PSPDF_ZjRptCONa, self.Q_PSPDF_ZjRptCONr ) # I(zPSD;CON|Zj)
            I_fcPE_ZjCr_ = MeanKLD(self.Q_PDPSD_ZjRptCONr, self.Q_PDPSD_Batch) # I(fcPE;CONr,Zj)
            I_fcPE_CaZj_ = MeanKLD(self.Q_PDPSD_ZjRptCONa, self.Q_PDPSD_ZjRptCONr) # I(fcPE;CONa,Zj)


            print('I_zPSD_Z :', I_zPSD_Z_)
            self.SubResDic['I_zPSD_Z'].append(I_zPSD_Z_)
            self.I_zPSD_Z += I_zPSD_Z_

            print('I_zPSD_ZjZ :', I_zPSD_ZjZ_)
            self.SubResDic['I_zPSD_ZjZ'].append(I_zPSD_ZjZ_)
            self.I_zPSD_ZjZ += I_zPSD_ZjZ_
            
            print('I_zPSD_ZjCr :', I_zPSD_ZjCr_)
            self.SubResDic['I_zPSD_ZjCr'].append(I_zPSD_ZjCr_)
            self.I_zPSD_ZjCr += I_zPSD_ZjCr_

            print('I_zPSD_CaZj :', I_zPSD_CaZj_)
            self.SubResDic['I_zPSD_CaZj'].append(I_zPSD_CaZj_) 
            self.I_zPSD_CaZj += I_zPSD_CaZj_

            print('I_fcPE_ZjCr :', I_fcPE_ZjCr_)
            self.SubResDic['I_fcPE_ZjCr'].append(I_fcPE_ZjCr_)
            self.I_fcPE_ZjCr += I_fcPE_ZjCr_

            print('I_fcPE_CaZj :', I_fcPE_CaZj_)
            self.SubResDic['I_fcPE_CaZj'].append(I_fcPE_CaZj_)
            self.I_fcPE_CaZj += I_fcPE_CaZj_



            ### --------------------------- Locating the candidate Z values that generate plausible signals ------------------------- ###
            # Calculating the entropies given the probability density function of the power spectral. 
            ## This indicates which frequency is most activated in the generated signal.
            EntH = -np.sum(self.SubPSPDF_ZjRptCONa * np.log(self.SubPSPDF_ZjRptCONa), axis=1).ravel()
            
            # Getting the maximum frequency given the PSD from SubPSPDF_ZjRptCONa.
            ## The 0 frequency is excluded as it represents the constant term; by adding 1 to the index, the frequency and index can be aligned to be the same.
            ## Return shape: (Batch_size, N_sample)
            MaxFreq = np.argmax(self.SubPSPDF_ZjRptCONa, axis=1).ravel() + 1
            
            self.LocCandZs ( MaxFreq, EntH, self.Samp_ZjRPT,  self.CON_Arange)

            # Restructuring TrackerCand
            ## item[0] contains frequency domains
            ## item[1] contains tracked Z values, 2nd data, and metrics
            self.TrackerCand = {item[0]: {'TrackZs': np.concatenate(self.TrackerCand_Temp[item[0]]['TrackZs']), 
                                          'TrackSecData': np.concatenate(self.TrackerCand_Temp[item[0]]['TrackSecData']), 
                                          'TrackMetrics': np.concatenate(self.TrackerCand_Temp[item[0]]['TrackMetrics'])} 
                                          for item in self.TrackerCand_Temp.items() if len(item[1]['TrackSecData']) > 0} 


        # Conducting the task iteration
        self.Iteration(TaskLogic)


        # MI(V;Zj,Z)
        self.I_zPSD_Z /= (self.TotalIterSize)
        self.AggResDic['I_zPSD_Z'].append(self.I_zPSD_Z)
        self.I_zPSD_ZjZ /= (self.TotalIterSize)
        self.AggResDic['I_zPSD_ZjZ'].append(self.I_zPSD_ZjZ)
        self.MI_zPSD_ZjZ = self.I_zPSD_Z + self.I_zPSD_ZjZ             
        self.AggResDic['MI_zPSD_ZjZ'].append(self.MI_zPSD_ZjZ)

        # MI(V;Cr,Zj)
        self.I_zPSD_ZjCr /= (self.TotalIterSize)
        self.AggResDic['I_zPSD_ZjCr'].append(self.I_zPSD_ZjCr)
        self.I_zPSD_CaZj /= (self.TotalIterSize)
        self.AggResDic['I_zPSD_CaZj'].append(self.I_zPSD_CaZj)
        self.MI_zPSD_CrZj = self.I_zPSD_ZjCr + self.I_zPSD_CaZj       
        self.AggResDic['MI_zPSD_CrZj'].append(self.MI_zPSD_CrZj)

        # MI(VE;Ca,Zj) - MI(VE;Cr,Zj)
        self.I_fcPE_ZjCr /= (self.TotalIterSize)
        self.AggResDic['I_fcPE_ZjCr'].append(self.I_fcPE_ZjCr)
        self.I_fcPE_CaZj /= (self.TotalIterSize)
        self.AggResDic['I_fcPE_CaZj'].append(self.I_fcPE_CaZj)
        self.MI_fcPE_CaCr = self.I_fcPE_CaZj - self.I_fcPE_ZjCr
        self.AggResDic['MI_fcPE_CaCr'].append(self.MI_fcPE_CaCr)
