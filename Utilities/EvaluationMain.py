import numpy as np
from scipy.stats import mode
import itertools
import pickle
from tqdm import trange, tqdm

import tensorflow as tf
from tensorflow.keras import Model

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Utilities.AncillaryFunctions import FFT_PSD, ProbPermutation, MeanKLD, Sampler, SamplingZ, SamplingZj, SamplingFCs
from Utilities.Utilities import CompResource

       
class Evaluator ():
    
    def __init__ (self, MinFreq=1, MaxFreq=51,  SimSize = 1, NMiniBat=100,  NSubGen=100, NParts=5, ReparaStdZj = 1, NSelZ = 1, 
                  SampBatchSize = 1000, GenBatchSize = 1000, SelMetricCut = 1., SelMetricType = 'KLD', GPU=False, Name=None):

        
        # Optional parameters with default values
        self.MinFreq = MinFreq               # The minimum frequency value within the analysis range (default = 1).
        self.MaxFreq = MaxFreq               # The maximum frequency value within the analysis range (default = 51).
        self.SimSize = SimSize               # Then umber of simulation repetitions for aggregating metrics (default: 1)
        self.NMiniBat = NMiniBat             # The size of the mini-batch, splitting the task into N pieces of size NMiniBat.
        self.NSubGen = NSubGen                 # The number of generations (i.e., samplings) within a sample.
        self.NParts = NParts                 # The number of partitions (i.e., samplings) in generations within a sample.
        self.ReparaStdZj = ReparaStdZj       # The size of the standard deviation when sampling Zj (Samp_Zjb ~ N(0, ReparaStdZj)).
        self.NSelZ = NSelZ                   # The size of js to be selected at the same time (default: 1).
        self.SampBatchSize = SampBatchSize   # The batch size during prediction of the sampling model.
        self.GenBatchSize= GenBatchSize      # The batch size during prediction of the generation model.
        self.GPU = GPU                       # GPU vs CPU during model predictions (i.e., for SampModel and GenModel). "The CPU is strongly recommended for optimal precision."
        self.SelMetricCut = SelMetricCut     # The threshold for Zs and ancillary data where the metric value is below SelMetricCut.
        self.SelMetricType = SelMetricType   # The type of metric used for selecting Zs and ancillary data. 
        self.Name = Name                     # Model name.
        self.NGen = NSubGen * NParts          # The number of generations (i.e., samplings) within the mini-batch.
    
    
    ''' ------------------------------------------------------ Ancillary Functions ------------------------------------------------------'''

    ### ----------- Searching for candidate Zj for plausible signal generation ----------- ###
    def LocCandZsMaxFreq (self, CandQV, Samp_Z, SecData=None):
        # Shape of CandQV: (NMiniBat, N_frequency, NGen)
        # Shape of Samp_Z: (NMiniBat x NGen, LatDim)
        # Shape of SecData: (NMiniBat x NGen, SecDataDim)

        
        if self.SelMetricType == 'Entropy': 
            # Calculating the entropies given the probability density function of the power spectral.
            ## Return shape: (NMiniBat x NGen )
            Score = -np.sum(CandQV * np.log(CandQV), axis=1).ravel()
    
        elif self.SelMetricType == 'KLD':
            # Calculating KLD(QV_Batch||CandQV_T) (KLD_BatGen) and selecing IDs for which KLD_BatGen less than SelMetricCut.
            ## Shape of CandQV_T: (NMiniBat, N_frequency, NGen) -> (NMiniBat x NGen, N_frequency, 1), Shape of QV_Batch: (1, N_frequency, NMiniBat)
            CandQV_T = CandQV.transpose(0,2,1).reshape(self.NMiniBat*self.NGen, -1)[:,:,None]
            KLD_BatGen = np.sum(self.QV_Batch * np.log(self.QV_Batch / CandQV_T ), axis=1)
        
            ## Return shape: (NMiniBat x NGen )
            Score = np.min(KLD_BatGen, axis=-1)

        
        # Getting the maximum frequency given the PSD from CandQV.
        ## The 0 frequency is excluded as it represents the constant term; by adding 1 to the index, the frequency and index can be aligned to be the same.
        ## Return shape: (NMiniBat, NGen) -> (NMiniBat x NGen) for the computational efficiency (i.e, ravel function applied)
        MaxFreq = np.argmax(CandQV, axis=1).ravel() + 1

        for Freq, _ in self.BestZsMetrics.items():
            FreqIdx = np.where(MaxFreq == Freq)[0]

            # Skipping the remainder of the code if there are no FreqIdx present at the predefined frequencies.
            if len(FreqIdx) <1: 
                continue;

            # Calculating the minimum of Score and selecting candidate Z-values(CandZs)
            MinScoreIdx = np.argmin(Score[FreqIdx]) 
            MinScore = np.min(Score[FreqIdx]) 
            CandZs = Samp_Z[[FreqIdx[MinScoreIdx]]]
            
            # Tracking results
            self.TrackerCand_Temp[Freq]['TrackZs'].append(CandZs[None])
            self.TrackerCand_Temp[Freq]['TrackMetrics'].append(MinScore[None])
            
            if SecData is not None: # for processing secondary data (SecData).
                CandSecData = SecData[[FreqIdx[MinScoreIdx]]]
                self.TrackerCand_Temp[Freq]['TrackSecData'].append(CandSecData[None])
            else:
                CandSecData = None

            # Updating the Min_SumH value if the current iteration value is smaller.
            if MinScore < self.BestZsMetrics[Freq][0]:
                self.BestZsMetrics[Freq] = [MinScore, CandZs, CandSecData]
                print('Candidate Z updated! ', 'Freq:', Freq, ', Score:', np.round(MinScore, 4))


    
    ### ------------------------  # Selecting nested Z values and secondary data matrix --------------------- ###
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
                        if TrackMetrics < self.SelMetricCut }

        else:
            Results = {next(Cnt):{ 'TrackZs' : TrackZs, 'TrackSecData' : TrackSecData} 
                        for TrackZs, TrackSecData, TrackMetrics 
                        in zip(SubTrackerCand['TrackZs'], SubTrackerCand['TrackSecData'], SubTrackerCand['TrackMetrics'])
                        if TrackMetrics < self.SelMetricCut }
            
        return Results
    
    
    
    ### ------------------------------ Conducting task iteration ------------------------------ ###
    def Iteration (self, TaskLogic):

        # Just functional code for setting the initial position of the progress bar 
        self.StartBarPoint = self.TotalIterSize*(self.iter/self.TotalIterSize) 
        with trange(self.iter, self.TotalIterSize , initial=self.StartBarPoint, leave=False) as t:

            for sim in range(self.sim, self.SimSize):
                self.sim = sim

                # Check the types of ancillary data fed into the sampler model and define the pipeline accordingly.
                if self.SecDataType == 'CONDIN' : 
                    SplitData = [np.array_split(sub, self.SubIterSize) for sub in (self.AnalSig, self.TrueCond)] 

                else: # For models with a single input such as VAE and TCVAE.
                    SplitData = np.array_split(self.AnalSig, self.SubIterSize)    

                for mini in range(self.mini, self.SubIterSize):
                    self.mini = mini
                    self.iter += 1
                    print()

                    # Core part; the task logic as the function
                    if self.SecDataType  == 'CONDIN': 
                        TaskLogic([subs[mini] for subs in SplitData])
                    else:
                        TaskLogic(SplitData[mini])

                    t.update(1)
    
    
    
    ### ------------------- Selecting post-sampled Z values for generating plausible signals ------------------- ###
    def SelPostSamp (self, SelMetricCut=np.inf, BestZsMetrics=None, TrackerCand=None, SavePath=None ):
        
        ## Optional parameters
        # Updating The threshold value for selecting Zs in SubNestedZFix if needed.
        self.SelMetricCut = SelMetricCut

        # Setting arguments
        BestZsMetrics = self.BestZsMetrics if BestZsMetrics is None else BestZsMetrics
        TrackerCand = self.TrackerCand if TrackerCand is None else TrackerCand
                
        # Exploring FreqIDs available for signal generation  
        ## FreqIDs such as [9, 10, 11 ..... 45]
        ## item[0] contains frequency domains
        ## item[1][0] contains metrics
        self.CandFreqIDs = [item[0] for item in BestZsMetrics.items() if item[1][0] != np.inf ]
            
        
        # Selecting nested Z values and secondary data matrix
        '''  Constructing the dictionary : {'FreqID' : {'SubKeys' : { 'TrackZs' : Zs, 'TrackSecData' : Secondary-data }}}
        
           - Outermost Dictionary:
             - Key (FreqID): Represents frequency identifiers.
             - Value: A second-level dictionary (explained below).
        
           - Second-level Dictionary:
             - Key (SubKeys): Represents sub-key (i.e., ID) for the given 'FreqID'.
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

        
        # Saving intermedicate results into the hard disk
        if SavePath is not None:
            with open(SavePath, 'wb') as handle:
                pickle.dump(self.PostSamp, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return self.PostSamp
    
    
        
    ### -------------- Evaluating the KLD between the PSD of the true signals and the generated signals ---------------- ###
    def KLD_TrueGen (self, PostSamp=None, AnalSig=None, SecDataType=None, PlotDist=True): # Filtering Quality Index
    
        ## Required parameters
        # PostSamp: The post-sampled data for generating signals with the shape of ({'FreqID': {'SubKeys': {'TrackZs': Zs, 'TrackSecData': Secondary-data}}}).
        # AnalSig: The raw true signals for obtaining the population PSD.  Data shape: (N_PostSamp, N_Obs)
        
        ## Optional parameters
        # SecDataType: The ancillary data-type: Use 'FCIN' for FC values or 'CONDIN' for conditional inputs such as power spectral density.
        # PostSamp: The selected sampled data.

        
        # Setting arguments
        PostSamp = self.PostSamp if PostSamp is None else PostSamp
        AnalSig = self.AnalSig if AnalSig is None else AnalSig
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
        if SecDataType == 'FCIN':
            Data = [PostSecDataList[:, :2], PostSecDataList[:, 2:], PostZsList]
        elif SecDataType == 'CONDIN':  
             Data = [PostZsList, PostSecDataList]
        elif SecDataType == False :
             Data = PostZsList
            
          
        # Generating signals
        ## Return shape of data: (N_PostSamp, SigDim)
        self.GenSamp = CompResource (self.GenModel, Data, BatchSize=self.GenBatchSize, GPU=self.GPU)
            

        # Calculating the KLD between the PSD of the true signals and the generated signals    
        PSDGenSamp =  FFT_PSD(self.GenSamp, 'All', MinFreq = self.MinFreq, MaxFreq = self.MaxFreq)
        PSDTrueData =  FFT_PSD(AnalSig, 'All', MinFreq = self.MinFreq, MaxFreq = self.MaxFreq)
            
            
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
    def Eval_ZFC (self, AnalSig, SampZModel, SampFCModel, GenModel,  FcLimit=0.05,  WindowSize=3,  SecDataType='FCIN',  Continue=True ):
        
        ## Required parameters
        self.AnalSig = AnalSig              # The data to be used for analysis.
        self.SampZModel = SampZModel        # The model that samples Zs.
        self.SampFCModel = SampFCModel      # The model that samples FCs.
        self.GenModel = GenModel            # The model that generates signals based on given Zs and FCs.
        
        assert SecDataType in ['FCIN','CONDIN', False], "Please verify the value of 'SecDataType'. Only 'FCIN', 'CONDIN'  or False are valid."
        
        
        ## Optional parameters with default values ##
        # WindowSize: The window size when calculating the permutation sets (default: 3)
        # Continue: Start from the beginning (Continue = False) vs. Continue where left off (Continue = True)
        self.FcLimit = FcLimit           # The threshold value of the max of the FC value input into the generation model (default: 0.05, i.e., frequency 5 Hertz)      
        self.SecDataType = SecDataType   # The ancillary data-type: Use 'FCIN' for FC values or 'CONDIN' for conditional inputs such as power spectral density.
        
        
        
        ## Intermediate variables
        self.Ndata = len(AnalSig) # The dimension size of the data.
        self.NFCs = GenModel.get_layer('Inp_FCEach').output.shape[-1] + GenModel.get_layer('Inp_FCCommon').output.shape[-1] # The dimension size of FCs.
        self.LatDim = SampZModel.output.shape[-1] # The dimension size of Z.
        self.SigDim = AnalSig.shape[-1] # The dimension (i.e., length) size of the raw signal.
        self.SubIterSize = self.Ndata//self.NMiniBat
        self.TotalIterSize = self.SubIterSize * self.SimSize
        
        
        # Functional trackers
        if Continue == False or not hasattr(self, 'iter'):
            self.sim, self.mini, self.iter = 0, 0, 0
        
            ## Result trackers
            self.SubResDic = {'I_V_Z':[],'I_V_ZjZ':[],'I_V_Zj':[],'I_V_FCsZj':[],'I_S_Zj':[],'I_S_FCsZj':[]}
            self.AggResDic = {'I_V_Z':[],'I_V_ZjZ':[],'I_V_Zj':[],'I_V_FCsZj':[],'I_S_Zj':[],'I_S_FCsZj':[], 
                         'MI_V_ZjZ':[], 'MI_V_FCsZj':[], 'MI_S_FCsZj':[]}
            self.BestZsMetrics = {i:[np.inf] for i in range(1, self.MaxFreq - self.MinFreq + 2)}
            self.TrackerCand_Temp = {i:{'TrackSecData':[],'TrackZs':[],'TrackMetrics':[] } for i in range(1, self.MaxFreq - self.MinFreq + 2)} 
            self.I_V_Z, self.I_V_ZjZ, self.I_V_Zj, self.I_V_FCsZj, self.I_S_Zj, self.I_S_FCsZj = 0,0,0,0,0,0
        
         

        
        ### ------------------------------------------------ Task logics ------------------------------------------------ ###
        
        # P(V=v)
        ## Data shape: (N_frequency)
        self.QV_Pop = FFT_PSD(self.AnalSig, 'All', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)
        
        
        def TaskLogic(SubData):

            print('-------------  ',self.Name,'  -------------')

            ### ------------------------------------------------ Sampling ------------------------------------------------ ###
            # Updating NMiniBat; If there is a remainder in Ndata/NMiniBat, NMiniBat must be updated." 
            self.NMiniBat = len(SubData) 

            
            # Sampling Samp_Z and Samp_Zj
            # Please note that the tensor is maintained in a reduced number of dimensions for computational efficiency in practice.
            ## Dimensionality Mapping in Our Paper: b: skipped, d: NMiniBat, r: NParts, m: NSubGen, j: LatDim; 
            # The values of z are randomly sampled at dimensions b, d, r, and j, while remaining constant across dimension m.
            self.Zbdr = SamplingZ(SubData, self.SampZModel, self.NMiniBat, self.NParts, self.NSubGen, 
                                BatchSize = self.SampBatchSize, GPU=self.GPU, SampZType='Modelbdr', ReparaStdZj=self.ReparaStdZj)
            self.Zbdr_Ext = self.Zbdr.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            
            # The values of z are randomly sampled at dimensions b, d, and j, while remaining constant across dimensions r and m.
            self.Zbd = np.broadcast_to(self.Zbdr_Ext[:,0,0][:,None,None], (self.NMiniBat, self.NParts, self.NSubGen, self.LatDim)).reshape(-1, self.LatDim)
            
            # Selecting Samp_Zjs from Zbd 
            self.Zjbd = SamplingZj (self.Zbd, self.NMiniBat, self.NParts, self.NSubGen, self.LatDim, self.NSelZ, ZjType='bd' ).copy()
            
            # Selecting sub-Zjbd from Zjbd for I_V_FCsZj
            self.Zjbd_Ext = self.Zjbd.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            # Return shape of Zjbd_Red1 : (NMiniBat*NSubGen, LatDim)
            ## The values of z are randomly sampled at dimensions b, d, and j, while remaining constant across dimension m.
            self.Zjbd_Red1 = self.Zjbd_Ext[:, 0].reshape(self.NMiniBat*self.NSubGen, -1).copy()
            # Return shape of Zjbd_Red2 : (NMiniBat, LatDim)
            ## The values of z are randomly sampled at dimensions b, d, and j.
            self.Zjbd_Red2 = self.Zjbd_Ext[:, 0, 0].copy()
            
            
            # Sampling Samp_FC
            ## Dimensionality Mapping in Our Paper: b: skipped, d: NMiniBat, r: NParts, m: NSubGen, k: LatDim; 
            # The values of FC are randomly sampled across all dimensions b, d, r, m, and k.
            self.FCbdrm = SamplingFCs (SubData, self.SampFCModel, self.NMiniBat, self.NParts, self.NSubGen, 
                                      BatchSize = self.SampBatchSize, GPU=self.GPU, SampFCType='Modelbdrm', FcLimit= self.FcLimit)
            self.FCbdrm_Ext = self.FCbdrm.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            
            # The values of FC are randomly sampled at the dimensions b, d, m, and k, and constant across dimension r.
            self.FCbdm = np.broadcast_to(self.FCbdrm_Ext[:, 0][:,None], (self.NMiniBat, self.NParts, self.NSubGen, self.NFCs)).reshape(-1, self.NFCs)
            
            # Sorting the arranged FC values in ascending order at the generation index.
            self.FCbdm_Ext = self.FCbdm.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            # Return shape of FCbdm_Sort : (NMiniBat*NSubGen, LatDim)
            ## The values of FC are sorted at the generation index after being randomly sampled across the dimensions b, d, m, and k.
            self.FCbdm_Sort = np.sort(self.FCbdm_Ext , axis=2)[:,0].reshape(self.NMiniBat*self.NSubGen, self.NFCs)
            # Return shape of FCbd_Sort : (NMiniBat, LatDim)
            ## The values of FC are sorted at the dimension d after being randomly sampled across the dimensions b, d and k.
            self.FCbd_Sort = np.sort(self.FCbdm_Ext[:, 0, 0], axis=0).copy() 


            
            ### ------------------------------------------------ Signal reconstruction ------------------------------------------------ ###
            '''
            - To maximize the efficiency of GPU utilization, 
              we performed a binding operation transforming tensors to (NMiniBat * NParts * NSubGen, LatDim) for Zs or (NMiniBat * NParts * NSubGen, NFCs) for FCs. 
              After the computation, we then reverted them back to their original dimensions.
                       
                                        ## Variable cases for the signal generation ##
                    
              # Cases                             # Super Signal                    # Sub-Signal                # Target metric
              1) Zbdr + FCbdrm         ->         Sig_Zbdr_FCbdrm        ->         Sig_Zbd_FCbdm               I() // H() or KLD ()
              2) Zjbd + FCbdrm         ->         Sig_Zjbd_FCbdrm        ->         Sig_Zjbd_FCbdm              I() 
              3) Zjbd + FCbdm_Sort     ->         Sig_Zjbd_FCbdmSt       ->                                     I() 
              4) Zjbd + FCbd_Sort      ->         Sig_Zjbd_FCbdSt        ->                                     I()  
                                                  * St=Sort 
             '''

            # Binding the samples together, generate signals through the model 
            ListZs = [ self.Zbdr,   self.Zjbd,    self.Zjbd_Red1,       self.Zjbd_Red2]
            Set_Zs = np.concatenate(ListZs)            
            Set_FCs = np.concatenate([self.FCbdrm, self.FCbdrm,  self.FCbdm_Sort,  self.FCbd_Sort]) 
            Set_Data = [Set_FCs[:, :2], Set_FCs[:, 2:], Set_Zs]
            
            # Gneraing indices for Re-splitting predictions for each case
            CaseLens = np.array([item.shape[0] for item in ListZs])
            DataCaseIDX = [0] + list(np.cumsum(CaseLens))
            
            # Choosing GPU or CPU and generating signals
            Set_Pred = CompResource (self.GenModel, Set_Data, BatchSize=self.GenBatchSize, GPU=self.GPU)
            
            # Re-splitting predictions for each case
            self.Sig_Zbdr_FCbdrm, self.Sig_Zjbd_FCbdrm, self.Sig_Zjbd_FCbdmSt, self.Sig_Zjbd_FCbdSt  = [Set_Pred[DataCaseIDX[i]:DataCaseIDX[i+1]] for i in range(len(DataCaseIDX)-1)] 
            
            self.Sig_Zbdr_FCbdrm = self.Sig_Zbdr_FCbdrm.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            self.Sig_Zjbd_FCbdrm = self.Sig_Zjbd_FCbdrm.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            self.Sig_Zjbd_FCbdmSt = self.Sig_Zjbd_FCbdmSt.reshape(self.NMiniBat, self.NSubGen, -1)
            self.Sig_Zjbd_FCbdSt = self.Sig_Zjbd_FCbdSt.reshape(self.NMiniBat, -1)
            
            self.Sig_Zbd_FCbdm = self.Sig_Zbdr_FCbdrm[:, 0]
            self.Sig_Zjbd_FCbdm = self.Sig_Zjbd_FCbdrm[:, 0]



            ### ------------------------------------------------ Calculating metrics for the evaluation ------------------------------------------------ ###
            
            '''                                        ## Sub-Metric list ##
                ------------------------------------------------------------------------------------------------------------- 
                # Sub-metrics   # Function             # Code                           # Function            # Code 
                1) I_V_Z        q(v|Sig_Zbd_FCbdm)     <QV_Zbd_FCbdm>           vs      p(v)                  <QV_Pop>
                2) I_V_ZjZ      q(v|Sig_Zjbd_FCbdm)    <QV_Zjbd_FCbdm>          vs      q(v|Sig_Zbd_FCbdm)    <QV_Zbd_FCbdm>
                
                3) I_V_Zj       q(v|Sig_Zjbd_FCbdm)    <QV_Zjbd_FCbdm>          vs      p(v)                  <QV_Pop>
                4) I_V_FCsZj    q(v|Sig_Zjbd_FCbdSt)   <QV_Zjbd_FCbdSt>         vs      q(v|Sig_Zjbd_FCbdm)   <QV_Zjbd_FCbdm>
                
                5) I_S_Zj       q(s|Sig_Zjbd_FCbdrm)   <QV//QS_Zjbd_FCbdrm>     vs      p(s|Sig_Zbdr_FCbdrm)  <QV//QS_Zbdr_FCbdrm>
                6) I_S_FCsZj    q(s|Sig_Zjbd_FCbdmSt)  <QV//QS_Zjbd_FCbdmSt>    vs      q(s|Sig_Zjbd_FCbdrm)  <QV//QS_Zjbd_FCbdrm>
            
                7) H()//KLD()   q(v|Sig_Zbdr_FCbdrm)   <QV_Zbdr_FCbdrm>                 q(v)                  <QV_Batch>       
                
                                                       
                                                       ## Metric list ##
                --------------------------------------------------------------------------------------------------------------
                                                   - MI_V_ZjZ = I_V_Z + I_V_ZjZ
                                                   - MI_V_FCsZj = I_V_Zj + I_V_FCsZj
                                                   - MI_S_FCsZj = I_S_Zj + I_S_FCsZj
                                                   - H() or KLD()
                
            '''

      
            ### ---------------------------- Cumulative Power Spectral Density (PSD) over each frequency -------------------------------- ###
            # Return shape of MQV_Zbd_FCbdm : (NMiniBat, N_frequency)
            self.MQV_Zbd_FCbdm = FFT_PSD(self.Sig_Zbd_FCbdm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).mean(1)
            # Return shape of MQV_Zjbd_FCbdm : (NMiniBat, N_frequency)
            self.MQV_Zjbd_FCbdm = FFT_PSD(self.Sig_Zjbd_FCbdm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).mean(1)
            # Return shape of QV_Zjbd_FCbdSt : (NMiniBat, N_frequency)
            self.QV_Zjbd_FCbdSt = FFT_PSD(self.Sig_Zjbd_FCbdSt, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)[:,0]
            
            # Return shape of QV_Zbdr_FCbdrm : (NMiniBat, NParts, NSubGen, N_frequency)
            self.QV_Zbdr_FCbdrm = FFT_PSD(self.Sig_Zbdr_FCbdrm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)
            # Return shape of QV_Zjbd_FCbdrm : (NMiniBat, NParts, NSubGen, N_frequency)
            self.QV_Zjbd_FCbdrm = FFT_PSD(self.Sig_Zjbd_FCbdrm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)
            # Return shape of QV_Zjbd_FCbdmSt : (NMiniBat, NSubGen, N_frequency)
            self.QV_Zjbd_FCbdmSt = FFT_PSD(self.Sig_Zjbd_FCbdmSt, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)
            #print(self.MQV_Zbd_FCbdm.shape, self.MQV_Zjbd_FCbdm.shape, self.QV_Zjbd_FCbdSt.shape, self.QV_Zbdr_FCbdrm.shape, self.QV_Zjbd_FCbdrm.shape, self.QV_Zjbd_FCbdmSt.shape)
            
            ### ---------------------------- Permutation density given PSD over each generation -------------------------------- ###
            # Calculating PD-PSD over v and s.
            ## Return shape of QSV_Zbdr_FCbdrm : (NMiniBat, NParts, N_frequency, N_permutation_cases)
            self.QSV_Zbdr_FCbdrm = np.concatenate([ProbPermutation(self.QV_Zbdr_FCbdrm[:,i], WindowSize=WindowSize)[:,None] for i in range(self.NParts)], axis=1)
            ## Return shape of QSV_Zjbd_FCbdrm : (NMiniBat, NParts, N_frequency, N_permutation_cases)
            self.QSV_Zjbd_FCbdrm = np.concatenate([ProbPermutation(self.QV_Zjbd_FCbdrm[:,i], WindowSize=WindowSize)[:,None] for i in range(self.NParts)], axis=1)
            ## Return shape of QSV_Zjbd_FCbdmSt : (NMiniBat, N_frequency, N_permutation_cases)
            self.QSV_Zjbd_FCbdmSt = ProbPermutation(self.QV_Zjbd_FCbdmSt, WindowSize=WindowSize)
            
            # Marginalizing v to obtain PD-PSD(s).
            ## Return shape of QS_Zbdr_FCbdrm : (NMiniBat, NParts, N_permutation_cases)
            self.QS_Zbdr_FCbdrm = np.sum(self.QSV_Zbdr_FCbdrm, axis=2)
            ## Return shape of QS_Zjbd_FCbdrm : (NMiniBat, NParts, N_permutation_cases)
            self.QS_Zjbd_FCbdrm = np.sum(self.QSV_Zjbd_FCbdrm, axis=2)
            ## Return shape of QS_Zjbd_FCbdmSt : (NMiniBat, N_permutation_cases)
            self.QS_Zjbd_FCbdmSt = np.sum(self.QSV_Zjbd_FCbdmSt, axis=1)
            #print(self.QSV_Zbdr_FCbdrm.shape,  self.QSV_Zjbd_FCbdrm.shape, self.QSV_Zjbd_FCbdmSt.shape, self.QS_Zbdr_FCbdrm.shape, self.QS_Zjbd_FCbdrm.shape, self.QS_Zjbd_FCbdmSt.shape)
            
            # Averaging PD-PSD(s) over the Dimension m: Effect of Monte Carlo Simulation
            ## Return shape of MQS_Zjbd_FCbdrm : (NMiniBat, N_permutation_cases)
            self.MQS_Zjbd_FCbdrm = np.mean(self.QS_Zjbd_FCbdrm, axis=1)
            ## Return shape of MQS_Zbdr_FCbdrm : (NMiniBat, N_permutation_cases)
            self.MQS_Zbdr_FCbdrm = np.mean(self.QS_Zbdr_FCbdrm, axis=1)
            #print(self.MQS_Zjbd_FCbdrm.shape, self.MQS_Zbdr_FCbdrm.shape )
            
            
            ### ---------------------------------------- Mutual information ---------------------------------------- ###
            I_V_Z_ = MeanKLD(self.MQV_Zbd_FCbdm, self.QV_Pop[None] ) # I(V;z)
            I_V_ZjZ_ = MeanKLD(self.MQV_Zjbd_FCbdm, self.MQV_Zbd_FCbdm )  # I(V;z'|z)
            I_V_Zj_ =  MeanKLD(self.MQV_Zjbd_FCbdm, self.QV_Pop[None] ) # I(V;z')
            I_V_FCsZj_ = MeanKLD(self.QV_Zjbd_FCbdSt, self.MQV_Zjbd_FCbdm ) # I(V;fc'|z')
            I_S_Zj_ = MeanKLD(self.MQS_Zjbd_FCbdrm, self.MQS_Zbdr_FCbdrm ) # I(S;z')
            I_S_FCsZj_ = MeanKLD(self.QS_Zjbd_FCbdmSt, self.MQS_Zjbd_FCbdrm ) # I(S;fc'|z')


            print('I(V;z) :', I_V_Z_)
            self.SubResDic['I_V_Z'].append(I_V_Z_)
            self.I_V_Z += I_V_Z_

            print("I(V;z'|z) :", I_V_ZjZ_)
            self.SubResDic['I_V_ZjZ'].append(I_V_ZjZ_)
            self.I_V_ZjZ += I_V_ZjZ_

            print("I(V;z') :", I_V_Zj_)
            self.SubResDic['I_V_Zj'].append(I_V_Zj_)
            self.I_V_Zj += I_V_Zj_

            print("I(V;fc'|z') :", I_V_FCsZj_)
            self.SubResDic['I_V_FCsZj'].append(I_V_FCsZj_)
            self.I_V_FCsZj += I_V_FCsZj_

            print("I(S;z') :", I_S_Zj_)
            self.SubResDic['I_S_Zj'].append(I_S_Zj_)
            self.I_S_Zj += I_S_Zj_

            print("I(S;fc'|z') :", I_S_FCsZj_)
            self.SubResDic['I_S_FCsZj'].append(I_S_FCsZj_)
            self.I_S_FCsZj += I_S_FCsZj_
            
            
            ### --------------------------- Locating the candidate Z values that generate plausible signals ------------------------- ###
            ## Return shape: (1, N_frequency, NMiniBat)
            ### Since it is the true PSD, there are no M generations. 
            self.QV_Batch = FFT_PSD(SubData[:,None], 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).transpose((1,2,0))
            
            # Intermediate objects for Q(s) and H(')
            ## Return shape: (NMiniBat, N_frequency, NGen)
            #self.QV_Zjbd_FCbdrm_T = self.QV_Zjbd_FCbdrm.reshape(self.NMiniBat, self.NGen, -1).transpose(0,2,1)
            #self.LocCandZsMaxFreq ( self.QV_Zjbd_FCbdrm_T, self.Zjbd,  self.FCbdrm)
            self.QV_Zbdr_FCbdrm_T = self.QV_Zbdr_FCbdrm.reshape(self.NMiniBat, self.NGen, -1).transpose(0,2,1)
            self.LocCandZsMaxFreq ( self.QV_Zbdr_FCbdrm_T, self.Zbdr,  self.FCbdrm)
 
            # Restructuring TrackerCand
            ## item[0] contains frequency domains
            ## item[1] contains tracked Z values, 2nd data, and metrics
            self.TrackerCand = {item[0]: {'TrackZs': np.concatenate(self.TrackerCand_Temp[item[0]]['TrackZs']), 
                                          'TrackSecData': np.concatenate(self.TrackerCand_Temp[item[0]]['TrackSecData']), 
                                          'TrackMetrics': np.concatenate(self.TrackerCand_Temp[item[0]]['TrackMetrics'])} 
                                          for item in self.TrackerCand_Temp.items() if len(item[1]['TrackSecData']) > 0} 
            
            
        # Conducting the task iteration
        self.Iteration(TaskLogic)


        # MI(V;Z',Z)
        self.I_V_Z /= (self.TotalIterSize)
        self.AggResDic['I_V_Z'].append(self.I_V_Z)
        self.I_V_ZjZ /= (self.TotalIterSize)
        self.AggResDic['I_V_ZjZ'].append(self.I_V_ZjZ)
        self.MI_V_ZjZ = self.I_V_Z + self.I_V_ZjZ             
        self.AggResDic['MI_V_ZjZ'].append(self.MI_V_ZjZ)

        # MI(V;FC,Z')
        self.I_V_Zj /= (self.TotalIterSize)
        self.AggResDic['I_V_Zj'].append(self.I_V_Zj)
        self.I_V_FCsZj /= (self.TotalIterSize)
        self.AggResDic['I_V_FCsZj'].append(self.I_V_FCsZj)
        self.MI_V_FCsZj = self.I_V_Zj + self.I_V_FCsZj       
        self.AggResDic['MI_V_FCsZj'].append(self.MI_V_FCsZj)

        # MI(VE;FC',Z') 
        self.I_S_Zj /= (self.TotalIterSize)
        self.AggResDic['I_S_Zj'].append(self.I_S_Zj)
        self.I_S_FCsZj /= (self.TotalIterSize)
        self.AggResDic['I_S_FCsZj'].append(self.I_S_FCsZj)
        self.MI_S_FCsZj = self.I_S_Zj + self.I_S_FCsZj
        self.AggResDic['MI_S_FCsZj'].append(self.MI_S_FCsZj)


    
    
    
    
    ### -------------------------- Evaluating the performance of the model using only Z inputs  -------------------------- ###
    def Eval_Z (self, AnalSig, SampZModel, GenModel, FcLimit=0.05, WindowSize=3, Continue=True ):
        
        ## Required parameters
        self.AnalSig = AnalSig             # The data to be used for analysis.
        self.SampZModel = SampZModel           # The model that samples Zs.
        self.GenModel = GenModel             # The model that generates signals based on given Zs and FCs.
        self.SecDataType = False             # The ancillary data-type: False means there is no ancillary dataset. 
        
        ## Intermediate variables
        self.Ndata = len(AnalSig) # The dimension size of the data.
        self.LatDim = SampZModel.output.shape[-1] # The dimension size of Z.
        self.SigDim = AnalSig.shape[-1] # The dimension (i.e., length) size of the raw signal.
        self.SubIterSize = self.Ndata//self.NMiniBat
        self.TotalIterSize = self.SubIterSize * self.SimSize
        
        
        # Functional trackers
        if Continue == False or not hasattr(self, 'iter'):
            self.sim, self.mini, self.iter = 0, 0, 0
        
            ## Result trackers
            self.SubResDic = {'I_V_Z':[],'I_V_ZjZ':[]}
            self.AggResDic = {'I_V_Z':[],'I_V_ZjZ':[], 'MI_V_ZjZ':[]}
            self.BestZsMetrics = {i:[np.inf] for i in range(1, self.MaxFreq - self.MinFreq + 2)}
            self.TrackerCand_Temp = {i:{'TrackSecData':[],'TrackZs':[],'TrackMetrics':[] } for i in range(1, self.MaxFreq - self.MinFreq + 2)} 
            self.I_V_Z, self.I_V_ZjZ = 0,0
        
         

        
        ### ------------------------------------------------ Task logics ------------------------------------------------ ###
        
        # P(V=v)
        ## Data shape: (N_frequency)
        self.QV_Pop = FFT_PSD(self.AnalSig, 'All', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)
        
        
        def TaskLogic(SubData):

            ### ------------------------------------------------ Sampling ------------------------------------------------ ###
            # Updating NMiniBat; If there is a remainder in Ndata/NMiniBat, NMiniBat must be updated." 
            self.NMiniBat = len(SubData) 

            # Sampling Samp_Z and Samp_Zj
            # Please note that the tensor is maintained in a reduced number of dimensions for computational efficiency in practice.
            ## Dimensionality Mapping in Our Paper: b: skipped, d: NMiniBat; 
            # The values of z are randomly sampled at dimensions b, and d.
            self.Zbd = SamplingZ(SubData, self.SampZModel, self.NMiniBat, 1, 1, 
                                BatchSize = self.SampBatchSize, GPU=self.GPU, SampZType='Modelbd', ReparaStdZj=self.ReparaStdZj)
           
            # Selecting Samp_Zjs from Zbd 
            self.Zjbd = SamplingZj (self.Zbd, self.NMiniBat, 1, 1, self.LatDim, self.NSelZ, ZjType='bd' )

            
            

            ### ------------------------------------------------ Signal reconstruction ------------------------------------------------ ###
            '''
                                        ## Variable cases for the signal generation ##
              # Cases                     # Super Signal                   # Target metric
              1) Zbd           ->         Sig_Zbd              ->          I() // H() or KLD ()
              2) Zjbd          ->         Sig_Zjbd             ->          I() 
                                          
            ''' 
            
            # Binding the samples together, generate signals through the model 
            ListZs = [self.Zbd,    self.Zjbd]
            Set_Data = np.concatenate(ListZs)  

            # Gneraing indices for Re-splitting predictions for each case
            CaseLens = np.array([item.shape[0] for item in ListZs])
            DataCaseIDX = [0] + list(np.cumsum(CaseLens))
            
            # Choosing GPU or CPU and generating signals
            Set_Pred = CompResource (self.GenModel, Set_Data, BatchSize=self.GenBatchSize, GPU=self.GPU)

            # Re-splitting predictions for each case
            self.Sig_Zbd, self.Sig_Zjbd = [Set_Pred[DataCaseIDX[i]:DataCaseIDX[i+1]] for i in range(len(DataCaseIDX)-1)] 
            
            self.Sig_Zbd = self.Sig_Zbd.reshape(self.NMiniBat, -1)
            self.Sig_Zjbd = self.Sig_Zjbd.reshape(self.NMiniBat, -1)

            
            ### ------------------------------------------------ Calculating metrics for the evaluation ------------------------------------------------ ###
            
            '''                                        ## Sub-Metric list ##
                ------------------------------------------------------------------------------------------------------------- 
                # Sub-metrics     # Function       # Code                 # Function            # Code 
                1) I_V_Z          q(v|Sig_Zbd)     <QV_Zbd>       vs      p(v)                  <QV_Pop>
                2) I_V_ZjZ        q(v|Sig_Zjbd)    <QV_Zjbd>      vs      q(v|Sig_Zbd)          <QV_Zbd>
                3) H()//KLD()     q(v|Sig_Zbd)     <QV_Zbd>               q(v)                  <QV_Batch>       
                                                       
                                                       ## Metric list ##
                --------------------------------------------------------------------------------------------------------------
                                                   - MI_V_ZjZ = I_V_Z + I_V_ZjZ
                                                   - H() or KLD()
                
            '''
            
            
             ### ---------------------------- Cumulative Power Spectral Density (PSD) over each frequency -------------------------------- ###
            # Return shape of MQV_Zbd_FCbdm : (NMiniBat, N_frequency)
            self.QV_Zbd = FFT_PSD(self.Sig_Zbd, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).mean(1)
            # Return shape of MQV_Zjbd_FCbdm : (NMiniBat, N_frequency)
            self.QV_Zjbd = FFT_PSD(self.Sig_Zjbd, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).mean(1)


            ### ---------------------------------------- Mutual information ---------------------------------------- ###
            # zPSD and fcPE stand for z-wise power spectral density and fc-wise permutation sets, respectively.
            I_V_Z_ = MeanKLD(self.QV_Zbd, self.QV_Pop[None] ) # I(V;z)
            I_V_ZjZ_ = MeanKLD(self.QV_Zjbd, self.QV_Zbd )  # I(V;z'|z)
   


            print('I(V;z) :', I_V_Z_)
            self.SubResDic['I_V_Z'].append(I_V_Z_)
            self.I_V_Z += I_V_Z_

            print("I(V;z'|z) :", I_V_ZjZ_)
            self.SubResDic['I_V_ZjZ'].append(I_V_ZjZ_)
            self.I_V_ZjZ += I_V_ZjZ_


            ### --------------------------- Locating the candidate Z values that generate plausible signals ------------------------- ###
            ## Return shape: (1, N_frequency, NMiniBat)
            ### Since it is the true PSD, there are no M generations. 
            self.QV_Batch = FFT_PSD(SubData[:,None], 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).transpose((1,2,0))
            
            # Intermediate objects for Q(s) and H(')
            ## Return shape: (NMiniBat, N_frequency, NGen)
            self.QV_Zbdr_T = self.QV_Zbd.reshape(self.NMiniBat, self.NGen, -1).transpose(0,2,1)
            self.LocCandZsMaxFreq ( self.QV_Zbdr_T, self.Zbd)
 
            # Restructuring TrackerCand
            ## item[0] contains frequency domains
            ## item[1] contains tracked Z values, 2nd data, and metrics
            self.TrackerCand = {item[0]: {'TrackZs': np.concatenate(self.TrackerCand_Temp[item[0]]['TrackZs']), 
                                   'TrackMetrics': np.concatenate(self.TrackerCand_Temp[item[0]]['TrackMetrics'])} 
                                     for item in self.TrackerCand_Temp.items() if len(item[1]['TrackZs']) > 0} 
            
            
        # Conducting the task iteration
        self.Iteration(TaskLogic)


        # MI(V;Z',Z)
        self.I_V_Z /= (self.TotalIterSize)
        self.AggResDic['I_V_Z'].append(self.I_V_Z)
        self.I_V_ZjZ /= (self.TotalIterSize)
        self.AggResDic['I_V_ZjZ'].append(self.I_V_ZjZ)
        self.MI_V_ZjZ = self.I_V_Z + self.I_V_ZjZ             
        self.AggResDic['MI_V_ZjZ'].append(self.MI_V_ZjZ)

        
        
        
        
        

    ### -------------------------- Evaluating the performance of the model using both Z and Conditions -------------------------- ###
    def Eval_ZCON (self, AnalData, SampZModel, GenModel, FcLimit=0.05,  WindowSize=3,  SecDataType=None,  Continue=True ):
        
        ## Required parameters
        self.SampZModel = SampZModel         # The model that samples Zs.
        self.GenModel = GenModel             # The model that generates signals based on given Zs and Cons.
        
        assert SecDataType in ['FCIN','CONDIN', False], "Please verify the value of 'SecDataType'. Only 'FCIN', 'CONDIN'  or False are valid."
        
        
        
        ## Optional parameters with default values ##
        # WindowSize: The window size when calculating the permutation sets (default: 3).
        # Continue: Start from the beginning (Continue = False) vs. Continue where left off (Continue = True).
        self.SecDataType = SecDataType   # The ancillary data-type: Use 'FCIN' for FC values or 'CONDIN' for conditional inputs such as power spectral density.
        
        

        ## Intermediate variables
        self.AnalSig = AnalData[0]  # The raw true signals to be used for analysis.
        self.TrueCond = AnalData[1] # The raw true PSD to be used for analysis.
        self.Ndata = len(self.AnalSig) # The dimension size of the data.
        self.LatDim = SampZModel.output.shape[-1] # The dimension size of Z.
        self.SigDim =  self.AnalSig.shape[-1] # The dimension (i.e., length) size of the raw signal.
        self.CondDim = self.TrueCond.shape[-1] # The dimension size of the conditional inputs.
        self.SubIterSize = self.Ndata//self.NMiniBat
        self.TotalIterSize = self.SubIterSize * self.SimSize
        
        assert self.NGen >= self.CondDim, "NGen must be greater than or equal to CondDim for the evaluation."

        
        # Functional trackers
        if Continue == False or not hasattr(self, 'iter'):
            self.sim, self.mini, self.iter = 0, 0, 0
        
            ## Result trackers
            self.SubResDic = {'I_V_Z':[],'I_V_ZjZ':[],'I_V_Zj':[],'I_V_CONsZj':[],'I_S_Zj':[],'I_S_CONsZj':[]}
            self.AggResDic = {'I_V_Z':[],'I_V_ZjZ':[],'I_V_Zj':[],'I_V_CONsZj':[],'I_S_Zj':[],'I_S_CONsZj':[], 
                         'MI_V_ZjZ':[], 'MI_V_CONsZj':[], 'MI_S_CONsZj':[]}
            self.BestZsMetrics = {i:[np.inf] for i in range(1, self.MaxFreq - self.MinFreq + 2)}
            self.TrackerCand_Temp = {i:{'TrackSecData':[],'TrackZs':[],'TrackMetrics':[] } for i in range(1, self.MaxFreq - self.MinFreq + 2)} 
            self.I_V_Z, self.I_V_ZjZ, self.I_V_Zj, self.I_V_CONsZj, self.I_S_Zj, self.I_S_CONsZj = 0,0,0,0,0,0
        
         

        
        ### ------------------------------------------------ Task logics ------------------------------------------------ ###
        
        # P(V=v)
        ## Data shape: (N_frequency)
        self.QV_Pop = FFT_PSD(self.AnalSig, 'All', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)
        
        
        def TaskLogic(SubData):

            print('-------------  ',self.Name,'  -------------')

            ### ------------------------------------------------ Sampling ------------------------------------------------ ###
            # Updating NMiniBat; If there is a remainder in Ndata/NMiniBat, NMiniBat must be updated." 
            self.NMiniBat = len(SubData[0]) 
            self.SubCond = SubData[1]

            
            # Sampling Samp_Z and Samp_Zj
            # Please note that the tensor is maintained in a reduced number of dimensions for computational efficiency in practice.
            ## Dimensionality Mapping in Our Paper: b: skipped, d: NMiniBat, r: NParts, m: NSubGen, j: LatDim; 
            # The values of z are randomly sampled at dimensions b, d, r, and j, while remaining constant across dimension m.
            self.Zbdr = SamplingZ(SubData, self.SampZModel, self.NMiniBat, self.NParts, self.NSubGen, SecDataType='CONDIN',
                                BatchSize = self.SampBatchSize, GPU=self.GPU, SampZType='Modelbdr', ReparaStdZj=self.ReparaStdZj)
            self.Zbdr_Ext = self.Zbdr.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            
            # The values of z are randomly sampled at dimensions b, d, and j, while remaining constant across dimensions r and m.
            self.Zbd = np.broadcast_to(self.Zbdr_Ext[:,0,0][:,None,None], (self.NMiniBat, self.NParts, self.NSubGen, self.LatDim)).reshape(-1, self.LatDim)
            
            # Selecting Samp_Zjs from Zbd 
            self.Zjbd = SamplingZj (self.Zbd, self.NMiniBat, self.NParts, self.NSubGen, self.LatDim, self.NSelZ, ZjType='bd' ).copy()
            
            # Selecting sub-Zjbd from Zjbd for I_V_FCsZj
            self.Zjbd_Ext = self.Zjbd.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            # Return shape of Zjbd_Red1 : (NMiniBat*NSubGen, LatDim)
            ## The values of z are randomly sampled at dimensions b, d, and j, while remaining constant across dimension m.
            self.Zjbd_Red1 = self.Zjbd_Ext[:, 0].reshape(self.NMiniBat*self.NSubGen, -1).copy()
            # Return shape of Zjbd_Red2 : (NMiniBat, LatDim)
            ## The values of z are randomly sampled at dimensions b, d, and j.
            self.Zjbd_Red2 = self.Zjbd_Ext[:, 0, 0].copy()
            
            
            # Processing Conditional information 
            ### Generating random indices for selecting true conditions
            RandSelIDXbdm = np.random.randint(0, self.TrueCond.shape[0], self.NMiniBat * self.NSubGen)
            RandSelIDXbdrm = np.random.randint(0, self.TrueCond.shape[0], self.NMiniBat * self.NParts* self.NSubGen)
            
            ### Selecting the true conditions using the generated indices
            # True conditions are randomly sampled at the dimensions b, d, m, and k, and constant across dimension r.
            self.CONbdm = self.TrueCond[RandSelIDXbdm]
            self.CONbdm = np.broadcast_to(self.CONbdm.reshape(self.NMiniBat, self.NSubGen, -1)[:,None], (self.NMiniBat, self.NParts, self.NSubGen, self.CondDim))
            
            # True conditions are randomly sampled across all dimensions b, d, r, m, and k.
            self.CONbdrm = self.TrueCond[RandSelIDXbdrm]
            
            
            # Sorting the arranged condition values in ascending order at the generation index.
            self.CONbdm_Ext = self.CONbdm.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            # Return shape of CONbdm_Sort : (NMiniBat*NSubGen, LatDim)
            ## The conditions are sorted at the generation index after being randomly sampled across the dimensions b, d, m, and k.
            self.CONbdm_Sort = np.sort(self.CONbdm_Ext , axis=2)[:,0].reshape(self.NMiniBat*self.NSubGen, self.CondDim)
            # Return shape of CONbd_Sort : (NMiniBat, LatDim)
            ## The conditions are sorted at the dimension d after being randomly sampled across the dimensions b, d and k.
            self.CONbd_Sort = np.sort(self.CONbdm_Ext[:, 0, 0], axis=0).copy() 

            

            ### ------------------------------------------------ Signal reconstruction ------------------------------------------------ ###
            '''
            - To maximize the efficiency of GPU utilization, 
              we performed a binding operation transforming tensors to (NMiniBat * NParts * NSubGen, LatDim) for Zs or (NMiniBat * NParts * NSubGen, CondDim) for CON. 
              After the computation, we then reverted them back to their original dimensions.
                       
                                        ## Variable cases for the signal generation ##
                    
              # Cases                             # Super Signal                    # Sub-Signal                # Target metric
              1) Zbdr + CONbdrm         ->         Sig_Zbdr_CONbdrm        ->        Sig_Zbd_CONbdm              I() // H() or KLD ()
              2) Zjbd + CONbdrm         ->         Sig_Zjbd_CONbdrm        ->        Sig_Zjbd_CONbdm             I() 
              3) Zjbd + CONbdm_Sort     ->         Sig_Zjbd_CONbdmSt       ->                                    I() 
              4) Zjbd + CONbd_Sort      ->         Sig_Zjbd_CONbdSt        ->                                    I()  
                                                  * St=Sort 
             '''

                        
            # Binding the samples together, generate signals through the model 
            ListZs = [ self.Zbdr,   self.Zjbd,    self.Zjbd_Red1,       self.Zjbd_Red2]
            Set_Zs = np.concatenate(ListZs)            
            Set_CONs = np.concatenate([self.CONbdrm, self.CONbdrm,  self.CONbdm_Sort,  self.CONbd_Sort]) 
            Set_Data = [Set_Zs, Set_CONs]
            
            # Gneraing indices for Re-splitting predictions for each case
            CaseLens = np.array([item.shape[0] for item in ListZs])
            DataCaseIDX = [0] + list(np.cumsum(CaseLens))
            
            # Choosing GPU or CPU and generating signals
            Set_Pred = CompResource (self.GenModel, Set_Data, BatchSize=self.GenBatchSize, GPU=self.GPU)
            
            # Re-splitting predictions for each case
            self.Sig_Zbdr_CONbdrm, self.Sig_Zjbd_CONbdrm, self.Sig_Zjbd_CONbdmSt, self.Sig_Zjbd_CONbdSt  = [Set_Pred[DataCaseIDX[i]:DataCaseIDX[i+1]] for i in range(len(DataCaseIDX)-1)] 
            
            self.Sig_Zbdr_CONbdrm = self.Sig_Zbdr_CONbdrm.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            self.Sig_Zjbd_CONbdrm = self.Sig_Zjbd_CONbdrm.reshape(self.NMiniBat, self.NParts, self.NSubGen, -1)
            self.Sig_Zjbd_CONbdmSt = self.Sig_Zjbd_CONbdmSt.reshape(self.NMiniBat, self.NSubGen, -1)
            self.Sig_Zjbd_CONbdSt = self.Sig_Zjbd_CONbdSt.reshape(self.NMiniBat, -1)
            
            self.Sig_Zbd_CONbdm = self.Sig_Zbdr_CONbdrm[:, 0]
            self.Sig_Zjbd_CONbdm = self.Sig_Zjbd_CONbdrm[:, 0]



            ### ------------------------------------------------ Calculating metrics for the evaluation ------------------------------------------------ ###



 
            
            ### ------------------------------------------------ Calculating metrics for the evaluation ------------------------------------------------ ###
            
            '''                                        ## Sub-Metric list ##
                ------------------------------------------------------------------------------------------------------------- 
                # Sub-metrics   # Function             # Code                           # Function            # Code 
                1) I_V_Z        q(v|Sig_Zbd_CONbdm)     <QV_Zbd_CONbdm>           vs      p(v)                  <QV_Pop>
                2) I_V_ZjZ      q(v|Sig_Zjbd_CONbdm)    <QV_Zjbd_CONbdm>          vs      q(v|Sig_Zbd_CONbdm)    <QV_Zbd_CONbdm>
                
                3) I_V_Zj       q(v|Sig_Zjbd_CONbdm)    <QV_Zjbd_CONbdm>          vs      p(v)                  <QV_Pop>
                4) I_V_CONsZj    q(v|Sig_Zjbd_CONbdSt)   <QV_Zjbd_CONbdSt>         vs      q(v|Sig_Zjbd_CONbdm)   <QV_Zjbd_CONbdm>
                
                5) I_S_Zj       q(s|Sig_Zjbd_CONbdrm)   <QV//QS_Zjbd_CONbdrm>     vs      p(s|Sig_Zbdr_CONbdrm)  <QV//QS_Zbdr_CONbdrm>
                6) I_S_CONsZj    q(s|Sig_Zjbd_CONbdmSt)  <QV//QS_Zjbd_CONbdmSt>    vs      q(s|Sig_Zjbd_CONbdrm)  <QV//QS_Zjbd_CONbdrm>
            
                7) H()//KLD()   q(v|Sig_Zbdr_CONbdrm)   <QV_Zbdr_CONbdrm>                 q(v)                  <QV_Batch>       
                
                                                       
                                                       ## Metric list ##
                --------------------------------------------------------------------------------------------------------------
                                                   - MI_V_ZjZ = I_V_Z + I_V_ZjZ
                                                   - MI_V_CONsZj = I_V_Zj + I_V_CONsZj
                                                   - MI_S_CONsZj = I_S_Zj + I_S_CONsZj
                                                   - H() or KLD()
                
            '''
            
            
            ### ---------------------------- Cumulative Power Spectral Density (PSD) over each frequency -------------------------------- ###
            # Return shape of MQV_Zbd_CONbdm : (NMiniBat, N_frequency)
            self.MQV_Zbd_CONbdm = FFT_PSD(self.Sig_Zbd_CONbdm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).mean(1)
            # Return shape of MQV_Zjbd_CONbdm : (NMiniBat, N_frequency)
            self.MQV_Zjbd_CONbdm = FFT_PSD(self.Sig_Zjbd_CONbdm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).mean(1)
            # Return shape of QV_Zjbd_CONbdSt : (NMiniBat, N_frequency)
            self.QV_Zjbd_CONbdSt = FFT_PSD(self.Sig_Zjbd_CONbdSt, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)[:,0]
            
            # Return shape of QV_Zbdr_CONbdrm : (NMiniBat, NParts, NSubGen, N_frequency)
            self.QV_Zbdr_CONbdrm = FFT_PSD(self.Sig_Zbdr_CONbdrm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)
            # Return shape of QV_Zjbd_CONbdrm : (NMiniBat, NParts, NSubGen, N_frequency)
            self.QV_Zjbd_CONbdrm = FFT_PSD(self.Sig_Zjbd_CONbdrm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)
            # Return shape of QV_Zjbd_CONbdmSt : (NMiniBat, NSubGen, N_frequency)
            self.QV_Zjbd_CONbdmSt = FFT_PSD(self.Sig_Zjbd_CONbdmSt, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)
            #print(self.MQV_Zbd_CONbdm.shape, self.MQV_Zjbd_CONbdm.shape, self.QV_Zjbd_CONbdSt.shape, self.QV_Zbdr_CONbdrm.shape, self.QV_Zjbd_CONbdrm.shape, self.QV_Zjbd_CONbdmSt.shape)
            
            ### ---------------------------- Permutation density given PSD over each generation -------------------------------- ###
            # Calculating PD-PSD over v and s.
            ## Return shape of QSV_Zbdr_CONbdrm : (NMiniBat, NParts, N_frequency, N_permutation_cases)
            self.QSV_Zbdr_CONbdrm = np.concatenate([ProbPermutation(self.QV_Zbdr_CONbdrm[:,i], WindowSize=WindowSize)[:,None] for i in range(self.NParts)], axis=1)
            ## Return shape of QSV_Zjbd_CONCbdrm : (NMiniBat, NParts, N_frequency, N_permutation_cases)
            self.QSV_Zjbd_CONbdrm = np.concatenate([ProbPermutation(self.QV_Zjbd_CONbdrm[:,i], WindowSize=WindowSize)[:,None] for i in range(self.NParts)], axis=1)
            ## Return shape of QSV_Zjbd_CONbdmSt : (NMiniBat, N_frequency, N_permutation_cases)
            self.QSV_Zjbd_CONbdmSt = ProbPermutation(self.QV_Zjbd_CONbdmSt, WindowSize=WindowSize)
            
            # Marginalizing v to obtain PD-PSD(s).
            ## Return shape of QS_Zbdr_CONbdrm : (NMiniBat, NParts, N_permutation_cases)
            self.QS_Zbdr_CONbdrm = np.sum(self.QSV_Zbdr_CONbdrm, axis=2)
            ## Return shape of QS_Zjbd_CONbdrm : (NMiniBat, NParts, N_permutation_cases)
            self.QS_Zjbd_CONbdrm = np.sum(self.QSV_Zjbd_CONbdrm, axis=2)
            ## Return shape of QS_Zjbd_CONbdmSt : (NMiniBat, N_permutation_cases)
            self.QS_Zjbd_CONbdmSt = np.sum(self.QSV_Zjbd_CONbdmSt, axis=1)
            #print(self.QSV_Zbdr_CONbdrm.shape,  self.QSV_Zjbd_CONbdrm.shape, self.QSV_Zjbd_CONbdmSt.shape, self.QS_Zbdr_CONbdrm.shape, self.QS_Zjbd_CONbdrm.shape, self.QS_Zjbd_CONbdmSt.shape)
            
            # Averaging PD-PSD(s) over the Dimension m: Effect of Monte Carlo Simulation
            ## Return shape of MQS_Zjbd_CONbdrm : (NMiniBat, N_permutation_cases)
            self.MQS_Zjbd_CONbdrm = np.mean(self.QS_Zjbd_CONbdrm, axis=1)
            ## Return shape of MQS_Zbdr_CONbdrm : (NMiniBat, N_permutation_cases)
            self.MQS_Zbdr_CONbdrm = np.mean(self.QS_Zbdr_CONbdrm, axis=1)
            #print(self.MQS_Zjbd_CONbdrm.shape, self.MQS_Zbdr_CONbdrm.shape )

            
                
            ### ---------------------------------------- Mutual information ---------------------------------------- ###
            I_V_Z_ = MeanKLD(self.MQV_Zbd_CONbdm, self.QV_Pop[None] ) # I(V;z)
            I_V_ZjZ_ = MeanKLD(self.MQV_Zjbd_CONbdm, self.MQV_Zbd_CONbdm )  # I(V;z'|z)
            I_V_Zj_ =  MeanKLD(self.MQV_Zjbd_CONbdm, self.QV_Pop[None] ) # I(V;z')
            I_V_CONsZj_ = MeanKLD(self.QV_Zjbd_CONbdSt, self.MQV_Zjbd_CONbdm ) # I(V;Con'|z')
            I_S_Zj_ = MeanKLD(self.MQS_Zjbd_CONbdrm, self.MQS_Zbdr_CONbdrm ) # I(S;z')
            I_S_CONsZj_ = MeanKLD(self.QS_Zjbd_CONbdmSt, self.MQS_Zjbd_CONbdrm ) # I(S;Con'|z')


            print('I(V;z) :', I_V_Z_)
            self.SubResDic['I_V_Z'].append(I_V_Z_)
            self.I_V_Z += I_V_Z_
            
            print("I(V;z'|z) :", I_V_ZjZ_)
            self.SubResDic['I_V_ZjZ'].append(I_V_ZjZ_)
            self.I_V_ZjZ += I_V_ZjZ_
            
            print("I(V;z') :", I_V_Zj_)
            self.SubResDic['I_V_Zj'].append(I_V_Zj_)
            self.I_V_Zj += I_V_Zj_
            
            print("I(V;Con'|z') :", I_V_CONsZj_)
            self.SubResDic['I_V_CONsZj'].append(I_V_CONsZj_)
            self.I_V_CONsZj += I_V_CONsZj_
            
            print("I(S;z') :", I_S_Zj_)
            self.SubResDic['I_S_Zj'].append(I_S_Zj_)
            self.I_S_Zj += I_S_Zj_
            
            print("I(S;Con'|z') :", I_S_CONsZj_)
            self.SubResDic['I_S_CONsZj'].append(I_S_CONsZj_)
            self.I_S_CONsZj += I_S_CONsZj_
                        
            
            ### --------------------------- Locating the candidate Z values that generate plausible signals ------------------------- ###
            ## Return shape: (1, N_frequency, NMiniBat)
            ### Since it is the true PSD, there are no M generations. 
            self.QV_Batch = FFT_PSD(SubData[0][:,None], 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).transpose((1,2,0))
            
            # Intermediate objects for Q(s) and H(')
            ## Return shape: (NMiniBat, N_frequency, NGen)
            self.QV_Zbdr_CONbdrm_T = self.QV_Zbdr_CONbdrm.reshape(self.NMiniBat, self.NGen, -1).transpose(0,2,1)
            self.LocCandZsMaxFreq ( self.QV_Zbdr_CONbdrm_T, self.Zbdr,  self.CONbdrm)
            
            # Restructuring TrackerCand
            ## item[0] contains frequency domains
            ## item[1] contains tracked Z values, 2nd data, and metrics
            self.TrackerCand = {item[0]: {'TrackZs': np.concatenate(self.TrackerCand_Temp[item[0]]['TrackZs']), 
                                          'TrackSecData': np.concatenate(self.TrackerCand_Temp[item[0]]['TrackSecData']), 
                                          'TrackMetrics': np.concatenate(self.TrackerCand_Temp[item[0]]['TrackMetrics'])} 
                                          for item in self.TrackerCand_Temp.items() if len(item[1]['TrackSecData']) > 0} 
            
            
        # Conducting the task iteration
        self.Iteration(TaskLogic)


        # MI(V;Z',Z)
        self.I_V_Z /= (self.TotalIterSize)
        self.AggResDic['I_V_Z'].append(self.I_V_Z)
        self.I_V_ZjZ /= (self.TotalIterSize)
        self.AggResDic['I_V_ZjZ'].append(self.I_V_ZjZ)
        self.MI_V_ZjZ = self.I_V_Z + self.I_V_ZjZ             
        self.AggResDic['MI_V_ZjZ'].append(self.MI_V_ZjZ)

        # MI(V;CON,Z')
        self.I_V_Zj /= (self.TotalIterSize)
        self.AggResDic['I_V_Zj'].append(self.I_V_Zj)
        self.I_V_CONsZj /= (self.TotalIterSize)
        self.AggResDic['I_V_CONsZj'].append(self.I_V_CONsZj)
        self.MI_V_CONsZj = self.I_V_Zj + self.I_V_CONsZj       
        self.AggResDic['MI_V_CONsZj'].append(self.MI_V_CONsZj)

        # MI(VE;CON',Z') 
        self.I_S_Zj /= (self.TotalIterSize)
        self.AggResDic['I_S_Zj'].append(self.I_S_Zj)
        self.I_S_CONsZj /= (self.TotalIterSize)
        self.AggResDic['I_S_CONsZj'].append(self.I_S_CONsZj)
        self.MI_S_CONsZj = self.I_S_Zj + self.I_S_CONsZj
        self.AggResDic['MI_S_CONsZj'].append(self.MI_S_CONsZj)