import numpy as np
from scipy.stats import mode
import itertools
from tqdm import trange, tqdm

import tensorflow as tf
from tensorflow.keras import Model

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Utilities.AncillaryFunctions import FFT_PSD, ProbPermutation, MeanKLD, Sampler, SamplingZ, SamplingZj, SamplingFCs
from Utilities.Utilities import CompResource

       
class Evaluator ():
    
    def __init__ (self, MinFreq=1, MaxFreq=51,  SimSize = 1, NMiniBat=100,  NGen=100, ReparaStdZj = 1, NSelZ = 1, 
                  SampBatchSize = 1000, GenBatchSize = 1000, SelMetricCut = 1., SelMetricType = 'KLD', GPU=False, Name=None):

        
        # Optional parameters with default values
        self.MinFreq = MinFreq               # The minimum frequency value within the analysis range (default = 1).
        self.MaxFreq = MaxFreq               # The maximum frequency value within the analysis range (default = 51).
        self.SimSize = SimSize               # Then umber of simulation repetitions for aggregating metrics (default: 1)
        self.NMiniBat = NMiniBat             # The size of the mini-batch, splitting the task into N pieces of size NMiniBat.
        self.NGen = NGen                     # The number of generations (i.e., samplings) within the mini-batch.
        self.ReparaStdZj = ReparaStdZj       # The size of the standard deviation when sampling Zj (Samp_Zjb ~ N(0, ReparaStdZj)).
        self.NSelZ = NSelZ                   # The size of js to be selected at the same time (default: 1).
        self.SampBatchSize = SampBatchSize   # The batch size during prediction of the sampling model.
        self.GenBatchSize= GenBatchSize      # The batch size during prediction of the generation model.
        self.GPU = GPU                       # GPU vs CPU during model predictions (i.e., for SampModel and GenModel). "The CPU is strongly recommended for optimal precision."
        self.SelMetricCut = SelMetricCut     # The threshold for Zs and ancillary data where the metric value is below SelMetricCut.
        self.SelMetricType = SelMetricType   # The type of metric used for selecting Zs and ancillary data. 
        self.Name = Name                     # Model name.

    
    
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

        
        return self.PostSamp
    
    
        
    ### -------------- Evaluating the KLD between the PSD of the true signals and the generated signals ---------------- ###
    def KLD_TrueGen (self, PostSamp=None, AnalSig=None, SecDataType=None, PlotDist=True):
    
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
    def Eval_ZFC (self, AnalSig, SampModel, GenModel, FcLimit=0.05,  WindowSize=3,  SecDataType='FCIN',  Continue=True ):
        
        ## Required parameters
        self.AnalSig = AnalSig             # The data to be used for analysis.
        self.SampModel = SampModel           # The model that samples Zs.
        self.GenModel = GenModel             # The model that generates signals based on given Zs and FCs.
        
        assert SecDataType in ['FCIN','CONDIN', False], "Please verify the value of 'SecDataType'. Only 'FCIN', 'CONDIN'  or False are valid."
        
        
        ## Optional parameters with default values ##
        # WindowSize: The window size when calculating the permutation sets (default: 3)
        # Continue: Start from the beginning (Continue = False) vs. Continue where left off (Continue = True)
        self.FcLimit = FcLimit           # The threshold value of the max of the FC value input into the generation model (default: 0.05, i.e., frequency 5 Hertz)      
        self.SecDataType = SecDataType   # The ancillary data-type: Use 'FCIN' for FC values or 'CONDIN' for conditional inputs such as power spectral density.
        
        
        
        ## Intermediate variables
        self.Ndata = len(AnalSig) # The dimension size of the data.
        self.NFCs = GenModel.get_layer('Inp_FCEach').output.shape[-1] + GenModel.get_layer('Inp_FCCommon').output.shape[-1] # The dimension size of FCs.
        self.LatDim = SampModel.output.shape[-1] # The dimension size of Z.
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

            # Sampling Samp_Z 
            ## The values are repeated NGen times after the sampling. 
            self.Zb = SamplingZ(SubData, self.SampModel, self.NMiniBat, self.NGen, 
                               BatchSize = self.SampBatchSize, GPU=self.GPU, SampZType='ModelBRpt', ReparaStdZj=self.ReparaStdZj)

            self.Zbm = SamplingZ(SubData, self.SampModel, self.NMiniBat, self.NGen, 
                               BatchSize = self.SampBatchSize, GPU=self.GPU, SampZType='ModelARand', ReparaStdZj=self.ReparaStdZj)
                        
            # Selecting Samp_Zjs from Samp_Z 
            ## For Samp_Zjs, the same j is selected in all generations within a mini-batch.
            self.Zjb = SamplingZj (self.Zb, self.NMiniBat, self.NGen, self.LatDim, self.NSelZ, ZjType='BRpt')

            # Sampling Samp_Zjs by fixing j            
            self.Zjbm = self.Zjb.copy()  
            # Replacing values where Zj is 0 with Zbm
            self.Zjbm[self.Zjb == 0] = self.Zbm[self.Zjb == 0]

            
            # Sampling FCs
            ## Return shape of FCs: (NMiniBat*NGen, NFCs) instead of (NMiniBat, NGen, NFCs) for optimal use of GPU
            # Generating FC values randomly across all axes (i.e., the batch, generation, and fc axes).
            self.FCbm = SamplingFCs(self.NMiniBat,  self.NGen, self.NFCs, SampFCType='ARand', FcLimit = self.FcLimit)
            # Generating FC values randomly across the batch and fc axes, and then repeat them NGen times.
            self.FCb = SamplingFCs(self.NMiniBat,  self.NGen, self.NFCs, SampFCType='BRpt', FcLimit = self.FcLimit)
            # Sorting the arranged FC values in ascending order at the generation index.
            self.FCbm_Sort = np.sort(self.FCbm , axis=0).reshape(self.NMiniBat*self.NGen, self.NFCs)


            

            ### ------------------------------------------------ Signal reconstruction ------------------------------------------------ ###
            '''
            - To maximize the efficiency of GPU utilization, 
              we performed a binding operation on (NMiniBat, NGen, LatDim) for Zs and (NMiniBat, NGen, NFCs) for FCs, respectively, 
              transforming them to (NMiniBat * NGen, LatDim) and (NMiniBat * NGen, NFCs). 
              After the computation, we then reverted them back to their original dimensions.
                       
                                ## Variable cases for the signal generation ##
                                
              # Cases                               # Signal name                   # Target metric
              1) Zb + FCbm               ->         Sig_Zb_FCbm         ->          MI() 
              2) Zjb + FCbm              ->         Sig_Zjb_FCbm        ->          MI()
              3) Zjb + FCbm_Sort         ->         Sig_Zjb_FCbmSt       ->          MI()
              4) Zjbm + FCbm             ->         Sig_Zjbm_FCbm       ->          H() or KLD()
                                                    * bm = ARand, b=BRpt, St=Sort *
            ''' 
            
            ## Binding the samples together, generate signals through the model 
            Set_Zs = np.concatenate([self.Zjb,  self.Zjb,        self.Zjbm])            
            Set_FCs = np.concatenate([self.FCbm, self.FCbm_Sort,  self.FCbm]) 
            Data = [Set_FCs[:, :2], Set_FCs[:, 2:], Set_Zs]


            # Choosing GPU or CPU and generating signals
            Set_Pred = CompResource (self.GenModel, Data, BatchSize=self.GenBatchSize, GPU=self.GPU)


            # Re-splitting predictions for each case
            Set_Pred = Set_Pred.reshape(-1, self.NMiniBat, self.NGen, self.SigDim)
            self.Sig_Zjb_FCbm, self.Sig_Zjb_FCbmSt, self.Sig_Zjbm_FCbm = [np.squeeze(SubPred) for SubPred in np.split(Set_Pred, 3)]  
            # Approximating Sig_Zb_FCbm using Sig_Zjbm_FCbm
            self.Sig_Zb_FCbm = self.Sig_Zjbm_FCbm.copy()


 

            ### ------------------------------------------------ Calculating metrics for the evaluation ------------------------------------------------ ###
            
            '''                                        ## Sub-Metric list ##
                ------------------------------------------------------------------------------------------------------------- 
                # Sub-metrics   # Function            # Code                       # Function           # Code 
                1) I_V_Z        q(v|Sig_Zb_FCbm)      <QV_Zb_FCbm>          vs     p(v)                 <QV_Pop>
                2) I_V_ZjZ      q(v|Sig_Zjb_FCbm)     <QV_Zjb_FCbm>         vs     q(v|Sig_Zb_FCbm)     <QV_Zb_FCbm>
                
                3) I_V_Zj       q(v|Sig_Zjb_FCbm)     <QV_Zjb_FCbm>         vs     p(v)                 <QV_Pop>
                4) I_V_FCsZj    q(v|Sig_Zjb_FCbmSt)   <QV_Zjb_FCbmSt>       vs     q(v|Sig_Zjb_FCbm)    <QV_Zjb_FCbm>
                
                5) I_S_Zj       q(s|Sig_Zjb_FCbm)     <QV//QS_Zjb_FCbm>     vs     p(s)                 <QV//QS_Batch>
                6) I_S_FCsZj    q(s|Sig_Zjb_FCbmSt)   <QV//QS_Zjb_FCbmSt>   vs     q(s|Sig_Zjb_FCbm)    <QV//QS_Zjb_FCbm>

                7) H()//KLD()   q(v|Sig_Zjbm_FCbm)    <QV_Zjbm_FCbm>       
                
                                                       
                                                       ## Metric list ##
                --------------------------------------------------------------------------------------------------------------
                                                   - MI_V_ZjZ = I_V_Z + I_V_ZjZ
                                                   - MI_V_FCsZj = I_V_Zj + I_V_FCsZj
                                                   - MI_S_FCsZj = I_S_Zj + I_S_FCsZj
                                                   - H() or KLD()
                
            '''
            
            
            ### ---------------------------- Cumulative Power Spectral Density (PSD) over each frequency -------------------------------- ###
            # Temporal objects with the shape : (NMiniBat, NGen, N_frequency)
            QV_Zjb_FCbm_ = FFT_PSD(self.Sig_Zjb_FCbm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)
            QV_Zjb_FCbmSt_ = FFT_PSD(self.Sig_Zjb_FCbmSt, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)
            

            # Q(v)s
            ## Return shape: (NMiniBat, N_frequency)
            self.QV_Zb_FCbm = FFT_PSD(self.Sig_Zb_FCbm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).mean(axis=1)
            self.QV_Zjb_FCbm = QV_Zjb_FCbm_.mean(axis=1)
            self.QV_Zjb_FCbmSt = QV_Zjb_FCbmSt_.mean(axis=1)

            
            # Intermediate objects for Q(s) and H(')
            ## Return shape: (NMiniBat, N_frequency, NGen)
            self.QV_Zjb_FCbm_T = QV_Zjb_FCbm_.transpose(0,2,1)
            self.QV_Zjb_FCbmSt_T = QV_Zjb_FCbmSt_.transpose(0,2,1)
            self.QV_Zjbm_FCbm_T = FFT_PSD(self.Sig_Zjbm_FCbm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).transpose(0,2,1)

            
            ## Return shape: (1, N_frequency, NMiniBat)
            ### Since it is the true PSD, there are no M generations. 
            self.QV_Batch = FFT_PSD(SubData[:,None], 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).transpose((1,2,0))


            ### ---------------------------- Permutation density given PSD over each generation -------------------------------- ###
            # Return shape: (NMiniBat, N_frequency, N_permutation_cases)
            self.QS_Zjb_FCbm = ProbPermutation(self.QV_Zjb_FCbm_T, WindowSize=WindowSize)
            self.QS_Zjb_FCbmSt = ProbPermutation(self.QV_Zjb_FCbmSt_T, WindowSize=WindowSize)
            self.QS_Batch = ProbPermutation(self.QV_Batch, WindowSize=WindowSize)

                
            ### ---------------------------------------- Mutual information ---------------------------------------- ###
            I_V_Z_ = MeanKLD(self.QV_Zb_FCbm, self.QV_Pop[None] ) # I(V;z)
            I_V_ZjZ_ = MeanKLD(self.QV_Zjb_FCbm, self.QV_Zb_FCbm )  # I(V;z'|z)
            I_V_Zj_ =  MeanKLD(self.QV_Zjb_FCbm, self.QV_Pop[None] ) # I(V;z')
            I_V_FCsZj_ = MeanKLD(self.QV_Zjb_FCbmSt, self.QV_Zjb_FCbm ) # I(V;fc'|z')
            I_S_Zj_ = MeanKLD(self.QS_Zjb_FCbm, self.QS_Batch) # I(S;z')
            I_S_FCsZj_ = MeanKLD(self.QS_Zjb_FCbmSt, self.QS_Zjb_FCbm) # I(S;fc'|z')


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
            self.LocCandZsMaxFreq ( self.QV_Zjbm_FCbm_T, self.Zjbm,  self.FCbm)
 
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
    def Eval_Z (self, AnalSig, SampModel, GenModel, FcLimit=0.05, WindowSize=3, Continue=True ):
        
        ## Required parameters
        self.AnalSig = AnalSig             # The data to be used for analysis.
        self.SampModel = SampModel           # The model that samples Zs.
        self.GenModel = GenModel             # The model that generates signals based on given Zs and FCs.
        self.SecDataType = False             # The ancillary data-type: False means there is no ancillary dataset. 
        
        ## Intermediate variables
        self.Ndata = len(AnalSig) # The dimension size of the data.
        self.LatDim = SampModel.output.shape[-1] # The dimension size of Z.
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

            print('-------------  ',self.Name,'  -------------')

            ### ------------------------------------------------ Sampling ------------------------------------------------ ###
            # Updating NMiniBat; If there is a remainder in Ndata/NMiniBat, NMiniBat must be updated." 
            self.NMiniBat = len(SubData) 

            # Sampling Samp_Z 
            ## The values are repeated NGen times after the sampling. 
            self.Zb = SamplingZ(SubData, self.SampModel, self.NMiniBat, self.NGen, 
                               BatchSize = self.SampBatchSize, GPU=self.GPU, SampZType='ModelBRpt', ReparaStdZj=self.ReparaStdZj)

            self.Zbm = SamplingZ(SubData, self.SampModel, self.NMiniBat, self.NGen, 
                               BatchSize = self.SampBatchSize, GPU=self.GPU, SampZType='ModelARand', ReparaStdZj=self.ReparaStdZj)
                        
            # Selecting Samp_Zjs from Samp_Z 
            ## For Samp_Zjs, the same j is selected in all generations within a mini-batch.
            self.Zjb = SamplingZj (self.Zb, self.NMiniBat, self.NGen, self.LatDim, self.NSelZ, ZjType='BRpt')

            # Sampling Samp_Zjs by fixing j               
            self.Zjbm = self.Zjb.copy()  
            # Replacing values where Zj is 0 with Zbm
            self.Zjbm[self.Zjb == 0] = self.Zbm[self.Zjb == 0]

            
            

            ### ------------------------------------------------ Signal reconstruction ------------------------------------------------ ###
            '''
            - To maximize the efficiency of GPU utilization, 
              we performed a binding operation on (NMiniBat, NGen, LatDim) for Zs and (NMiniBat, NGen, NFCs) for FCs, respectively, 
              transforming them to (NMiniBat X NGen, LatDim) and (NMiniBat X NGen, NFCs). 
              After the computation, we then reverted them back to their original dimensions.
                       
                                ## Variable cases for the signal generation ##
                                
              # Cases                         # Signal name                       # Target metric
              1) Zb                ->         Sig_Zb                 ->           MI() 
              2) Zjb               ->         Sig_Zjb                ->           MI()
              3) Zjbm              ->         Sig_Zjbm               ->           H() or KLD()
                                              * bm = ARand, b=BRpt, St=Sort *
            ''' 
            
            ## Binding the samples together, generate signals through the model 
            Data = np.concatenate([self.Zjb, self.Zjbm])            

            
            # Choosing GPU or CPU and generating signals
            Set_Pred = CompResource (self.GenModel, Data, BatchSize=self.GenBatchSize, GPU=self.GPU)


            # Re-splitting predictions for each case
            Set_Pred = Set_Pred.reshape(-1, self.NMiniBat, self.NGen, self.SigDim)
            self.Sig_Zjb, self.Sig_Zjbm = [np.squeeze(SubPred) for SubPred in np.split(Set_Pred, 2)]  
            # Approximating Sig_Zb_FCbm using Sig_Zjbm_FCbm
            self.Sig_Zb = self.Sig_Zjbm.copy()

            ### ------------------------------------------------ Calculating metrics for the evaluation ------------------------------------------------ ###
            
            '''                                        ## Sub-Metric list ##
                ------------------------------------------------------------------------------------------------------------- 
                # Sub-metrics    # Function        # Code               # Function           # Code 
                1) I_V_Z         q(v|Sig_Zb)      <QV_Zb>       vs      p(v)                 <QV_Pop>
                2) I_V_ZjZ       q(v|Sig_Zjb)     <QV_Zjb>      vs      q(v|Sig_Zb)          <QV_Zb>
                3) H()//KLD()    q(v|Sig_Zjbm)    <QV_Zjbm>       
                
                                                       
                                                       ## Metric list ##
                --------------------------------------------------------------------------------------------------------------
                                                   - MI_V_ZjZ = I_V_Z + I_V_ZjZ
                                                   - H() or KLD()                
            '''
            
            
            ### ---------------------------- Cumulative Power Spectral Density (PSD) over each frequency -------------------------------- ###

            # Q(v)s
            ## Return shape: (NMiniBat, N_frequency)
            self.QV_Zb = FFT_PSD(self.Sig_Zb, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).mean(axis=1)
            self.QV_Zjb = FFT_PSD(self.Sig_Zjb, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).mean(axis=1)
            
            
            # Intermediate objects for Q(s) and H(')
            ## Return shape: (NMiniBat, N_frequency, NGen)
            self.QV_Zjbm_T = FFT_PSD(self.Sig_Zjbm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).transpose(0,2,1)

            
            ## Return shape: (1, N_frequency, NMiniBat)
            ### Since it is the true PSD, there are no M generations. 
            self.QV_Batch = FFT_PSD(SubData[:,None], 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).transpose((1,2,0))


            ### ---------------------------------------- Mutual information ---------------------------------------- ###
            I_V_Z_ = MeanKLD(self.QV_Zb, self.QV_Pop[None] ) # I(V;z)
            I_V_ZjZ_ = MeanKLD(self.QV_Zjb, self.QV_Zb )  # I(V;z'|z)
   


            print('I(V;z) :', I_V_Z_)
            self.SubResDic['I_V_Z'].append(I_V_Z_)
            self.I_V_Z += I_V_Z_

            print("I(V;z'|z) :", I_V_ZjZ_)
            self.SubResDic['I_V_ZjZ'].append(I_V_ZjZ_)
            self.I_V_ZjZ += I_V_ZjZ_


            
            
            ### --------------------------- Locating the candidate Z values that generate plausible signals ------------------------- ###
            self.LocCandZsMaxFreq ( self.QV_Zjbm_T, self.Zjbm)
 
            # Restructuring TrackerCand
            ## item[0] contains frequency domains
            ## item[1] contains tracked Z values and metrics
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
    def Eval_ZCON (self, AnalData, SampModel, GenModel, FcLimit=0.05,  WindowSize=3,  SecDataType=None,  Continue=True ):
        
        ## Required parameters
        self.SampModel = SampModel           # The model that samples Zs.
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
        self.LatDim = SampModel.output.shape[-1] # The dimension size of Z.
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

            # Sampling Samp_Z 
            ## The values are repeated NGen times after the sampling. 
            self.Zb = SamplingZ(SubData, self.SampModel, self.NMiniBat, self.NGen, SecDataType='CONDIN',
                               BatchSize = self.SampBatchSize, GPU=self.GPU, SampZType='ModelBRpt', ReparaStdZj=self.ReparaStdZj)

            self.Zbm = SamplingZ(SubData, self.SampModel, self.NMiniBat, self.NGen, SecDataType='CONDIN',
                               BatchSize = self.SampBatchSize, GPU=self.GPU, SampZType='ModelARand', ReparaStdZj=self.ReparaStdZj)
                        
            # Selecting Samp_Zjs from Samp_Z 
            ## For Samp_Zjs, the same j is selected in all generations within a mini-batch.
            self.Zjb = SamplingZj (self.Zb, self.NMiniBat, self.NGen, self.LatDim, self.NSelZ, ZjType='BRpt')

            # Sampling Samp_Zjs by fixing j            
            self.Zjbm = self.Zjb.copy()  
            # Replacing values where Zj is 0 with Zbm
            self.Zjbm[self.Zjb == 0] = self.Zbm[self.Zjb == 0]


            
            # Processing Conditional information 
            ## Generating CONbm_Sort (NMiniBat x NGen, CondDim)
            ### Generating random indices for selecting true conditions
            RandSelIDX = np.random.randint(0, self.TrueCond.shape[0], self.NMiniBat * self.NGen)
            ### Selecting the true conditions using the generated indices
            SelTrueCond = self.TrueCond[RandSelIDX]
            ### Identifying the index of maximum frequency for each selected condition
            MaxFreqSelCond = np.argmax(SelTrueCond, axis=-1)
            ### Sorting the selected conditions by their maximum frequency
            Idx_MaxFreqSelCond = np.argsort(MaxFreqSelCond)
            self.CONbm_Sort = SelTrueCond[Idx_MaxFreqSelCond]


            ## Generating CONbm (NMiniBat x NGen, CondDim) by random Shuffling on both axes 
            ### Selecting elements (NMiniBat x NGen) randomly.
            self.CONbm = self.TrueCond[np.random.choice(self.Ndata, self.NMiniBat*self.NGen)]




            ### ------------------------------------------------ Signal reconstruction ------------------------------------------------ ###
            '''
            - To maximize the efficiency of GPU utilization, 
              we performed a binding operation on (NMiniBat, NGen, LatDim) for Zs and (NMiniBat, NGen, CondDim) for CONs, respectively, 
              transforming them to (NMiniBat * NGen, LatDim) and (NMiniBat * NGen, CondDim). 
              After the computation, we then reverted them back to their original dimensions.
                       
                                ## Variable cases for the signal generation ##
                                
              # Cases                               # Signal name                   # Target metric
              1) Zb + CONbm               ->         Sig_Zb_CONbm         ->          MI() 
              2) Zjb + CONbm              ->         Sig_Zjb_CONbm        ->          MI()
              3) Zjb + CONbm_Sort         ->         Sig_Zjb_CONbmSt      ->          MI()
              4) Zjbm + CONbm             ->         Sig_Zjbm_CONbm       ->          H() or KLD()
                                                    * bm = ARand, b=BRpt, St=Sort *
            ''' 
            
            ## Binding the samples together, generate signals through the model 
            Set_Zs = np.concatenate([self.Zjb,  self.Zjb,        self.Zjbm])            
            Set_CONs = np.concatenate([self.CONbm, self.CONbm_Sort,  self.CONbm]) 
            Data = [Set_Zs, Set_CONs]


            # Choosing GPU or CPU and generating signals
            Set_Pred = CompResource (self.GenModel, Data, BatchSize=self.GenBatchSize, GPU=self.GPU)


            # Re-splitting predictions for each case
            Set_Pred = Set_Pred.reshape(-1, self.NMiniBat, self.NGen, self.SigDim)
            self.Sig_Zjb_CONbm, self.Sig_Zjb_CONbmSt, self.Sig_Zjbm_CONbm = [np.squeeze(SubPred) for SubPred in np.split(Set_Pred, 3)]  
            # Approximating Sig_Zb_CONbm using Sig_Zjbm_CONbm
            self.Sig_Zb_CONbm = self.Sig_Zjbm_CONbm.copy()



 

            ### ------------------------------------------------ Calculating metrics for the evaluation ------------------------------------------------ ###
            
            '''                                        ## Sub-Metric list ##
                ------------------------------------------------------------------------------------------------------------- 
                # Sub-metrics   # Function            # Code                       # Function            # Code 
                1) I_V_Z        q(v|Sig_Zb_CONbm)      <QV_Zb_CONbm>          vs     p(v)                 <QV_Pop>
                2) I_V_ZjZ      q(v|Sig_Zjb_CONbm)     <QV_Zjb_CONbm>         vs     q(v|Sig_Zb_CONbm)    <QV_Zb_CONbm>
                
                3) I_V_Zj       q(v|Sig_Zjb_CONbm)     <QV_Zjb_CONbm>         vs     p(v)                 <QV_Pop>
                4) I_V_CONsZj   q(v|Sig_Zjb_CONbmSt)   <QV_Zjb_CONbmSt>       vs     q(v|Sig_Zjb_CONbm)   <QV_Zjb_CONbm>
                
                5) I_S_Zj       q(s|Sig_Zjb_CONbm)     <QV//QS_Zjb_CONbm>     vs     p(s)                 <QV//QS_Batch>
                6) I_S_CONsZj   q(s|Sig_Zjb_CONbmSt)   <QV//QS_Zjb_CONbmSt>   vs     q(s|Sig_Zjb_CONbm)   <QV//QS_Zjb_CONbm>

                7) H()//KLD()   q(v|Sig_Zjbm_CONbm)    <QV_Zjbm_CONbm>       
                
                                                       
                                                       ## Metric list ##
                --------------------------------------------------------------------------------------------------------------
                                                   - MI_V_ZjZ = I_V_Z + I_V_ZjZ
                                                   - MI_V_CONsZj = I_V_Zj + I_V_CONsZj
                                                   - MI_S_CONsZj = I_S_Zj + I_S_CONsZj
                                                   - H() or KLD()
                
            '''
            
            
            ### ---------------------------- Cumulative Power Spectral Density (PSD) over each frequency -------------------------------- ###
            # Temporal objects with the shape : (NMiniBat, NGen, N_frequency)
            QV_Zjb_CONbm_ = FFT_PSD(self.Sig_Zjb_CONbm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)
            QV_Zjb_CONbmSt_ = FFT_PSD(self.Sig_Zjb_CONbmSt, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq)
            

            # Q(v)s
            ## Return shape: (NMiniBat, N_frequency)
            self.QV_Zb_CONbm = FFT_PSD(self.Sig_Zb_CONbm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).mean(axis=1)
            self.QV_Zjb_CONbm = QV_Zjb_CONbm_.mean(axis=1)
            self.QV_Zjb_CONbmSt = QV_Zjb_CONbmSt_.mean(axis=1)

            
            # Intermediate objects for Q(s) and H(')
            ## Return shape: (NMiniBat, N_frequency, NGen)
            self.QV_Zjb_CONbm_T = QV_Zjb_CONbm_.transpose(0,2,1)
            self.QV_Zjb_CONbmSt_T = QV_Zjb_CONbmSt_.transpose(0,2,1)
            self.QV_Zjbm_CONbm_T = FFT_PSD(self.Sig_Zjbm_CONbm, 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).transpose(0,2,1)

            
            ## Return shape: (1, N_frequency, NMiniBat)
            ### Since it is the true PSD, there are no M generations. 
            self.QV_Batch = FFT_PSD(SubData[0][:,None], 'None', MinFreq=self.MinFreq, MaxFreq=self.MaxFreq).transpose((1,2,0))


            ### ---------------------------- Permutation density given PSD over each generation -------------------------------- ###
            # Return shape: (NMiniBat, N_frequency, N_permutation_cases)
            self.QS_Zjb_CONbm = ProbPermutation(self.QV_Zjb_CONbm_T, WindowSize=WindowSize)
            self.QS_Zjb_CONbmSt = ProbPermutation(self.QV_Zjb_CONbmSt_T, WindowSize=WindowSize)
            self.QS_Batch = ProbPermutation(self.QV_Batch, WindowSize=WindowSize)

                
            ### ---------------------------------------- Mutual information ---------------------------------------- ###
            I_V_Z_ = MeanKLD(self.QV_Zb_CONbm, self.QV_Pop[None] ) # I(V;z)
            I_V_ZjZ_ = MeanKLD(self.QV_Zjb_CONbm, self.QV_Zb_CONbm )  # I(V;z'|z)
            I_V_Zj_ =  MeanKLD(self.QV_Zjb_CONbm, self.QV_Pop[None] ) # I(V;z')
            I_V_CONsZj_ = MeanKLD(self.QV_Zjb_CONbmSt, self.QV_Zjb_CONbm ) # I(V;con'|z')
            I_S_Zj_ = MeanKLD(self.QS_Zjb_CONbm, self.QS_Batch) # I(S;z')
            I_S_CONsZj_ = MeanKLD(self.QS_Zjb_CONbmSt, self.QS_Zjb_CONbm) # I(S;con'|z')


            print('I(V;z) :', I_V_Z_)
            self.SubResDic['I_V_Z'].append(I_V_Z_)
            self.I_V_Z += I_V_Z_

            print("I(V;z'|z) :", I_V_ZjZ_)
            self.SubResDic['I_V_ZjZ'].append(I_V_ZjZ_)
            self.I_V_ZjZ += I_V_ZjZ_

            print("I(V;z') :", I_V_Zj_)
            self.SubResDic['I_V_Zj'].append(I_V_Zj_)
            self.I_V_Zj += I_V_Zj_

            print("I(V;con'|z') :", I_V_CONsZj_)
            self.SubResDic['I_V_CONsZj'].append(I_V_CONsZj_)
            self.I_V_CONsZj += I_V_CONsZj_

            print("I(S;z') :", I_S_Zj_)
            self.SubResDic['I_S_Zj'].append(I_S_Zj_)
            self.I_S_Zj += I_S_Zj_

            print("I(S;con'|z') :", I_S_CONsZj_)
            self.SubResDic['I_S_CONsZj'].append(I_S_CONsZj_)
            self.I_S_CONsZj += I_S_CONsZj_
            
            
            ### --------------------------- Locating the candidate Z values that generate plausible signals ------------------------- ###
            self.LocCandZsMaxFreq ( self.QV_Zjbm_CONbm_T, self.Zjbm,  self.CONbm)
 
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