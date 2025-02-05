from Benchmarks.Models.BaseVAES import *
from Benchmarks.Models.BenchmarkModels import *
from Utilities.Utilities import RelLossWeight
from Utilities.AncillaryFunctions import FFT_PSD

def ModelCall (ConfigSpec, ConfigName, TrData, ValData, Resume=False, LoadWeight=False, Reparam=True, ReparaStd=None, ModelSaveName=None):
    
    assert not (Resume or LoadWeight) or ModelSaveName is not None, "ModelSaveName must be provided to load the weights."
    DataSize = ConfigSpec['DataSize']
    SigDim = ConfigSpec['SigDim']
    
        
    # ModelName selection
    if 'BaseVAE' in ConfigName:
        BenchModel = BaseVAE(SigDim, ConfigSpec, Reparam=Reparam, ReparaStd=ReparaStd)
        TrInp, ValInp = TrData, ValData
    
    elif 'VDVAE' in ConfigName:
        BenchModel = VDVAE(SigDim, ConfigSpec, Reparam=Reparam, ReparaStd=ReparaStd)
        TrInp, ValInp = TrData, ValData
    
    elif 'TCVAE' in ConfigName:
        BenchModel = TCVAE(SigDim, DataSize, ConfigSpec, Reparam=Reparam, ReparaStd=ReparaStd)
        TrInp, ValInp = TrData, ValData
    
    elif 'FACVAE' in ConfigName:
        BenchModel = FACVAE(SigDim, ConfigSpec, Reparam=Reparam, ReparaStd=ReparaStd)
        TrInp, ValInp = TrData, ValData
    
    elif 'ConVAE' in ConfigName:
        ## Identifying conditions based on cumulative Power Spectral Entropy (PSE) over each frequency
        Tr_Cond = FFT_PSD(TrData, 'None')[:, 0]
        Val_Cond = FFT_PSD(ValData, 'None')[:, 0]            
        TrInp, ValInp = [TrData, Tr_Cond], [ValData, Val_Cond]
        CondDim = Tr_Cond.shape[-1]
        
        BenchModel= ConVAE(SigDim, CondDim, ConfigSpec, Reparam=Reparam, ReparaStd=ReparaStd)
    
    elif 'Wavenet' in ConfigName:
        SlidingSize = ConfigSpec['SlidingSize']
        TrSampled, TrRaw = TrData
        ValSampled, ValRaw = ValData
        ## Identifying 3-dimensional conditions using raw data, not sampled, based on PSE.
        Tr_Cond = FFT_PSD(TrRaw, 'None')[:, 0]
        Val_Cond = FFT_PSD(ValRaw, 'None')[:, 0]      
        TrInp, ValInp = [TrSampled, Tr_Cond], [ValSampled, Val_Cond]
        ConDim = Tr_Cond.shape[-1]

        # Call the model with some dummy input data to create the variables and allows weight loading
        BenchModel = Wavenet(SigDim, ConfigSpec, ConDim, SlidingSize = SlidingSize)
        BenchModel([Dummy[:1] for Dummy in TrInp])
        
    else:
        assert False, "Please verify if the model name is right or not."    

    
    # Model Training
    if Resume == True or LoadWeight == True:
        BenchModel.load_weights(ModelSaveName)
        print('Model weights loaded')
        
    return BenchModel, TrInp, ValInp



# Dynamic controller for losses
def DCLCall (ConfigSpec, ConfigName, ModelSaveName, ToSaveLoss=None, SaveWay='max', Buffer=0, Resume=False):
    
    if ToSaveLoss is None:
        ToSaveLoss = ['val_ReconOutLoss']

        
    ### Parameters for constant losse weights
    WRec = ConfigSpec['WRec']
    WZ = ConfigSpec['WZ']
    
    ### Parameters for dynamic controller for losse weights
    MnWRec = ConfigSpec['MnWRec']
    MnWZ = ConfigSpec['MnWZ']
    MxWRec = ConfigSpec['MxWRec']
    MxWZ = ConfigSpec['MxWZ']
        
        
    #### ------------------------------------------------ Dynamic controller for common losses and betas ------------------------------------------------ 
    RelLossDic = { 'val_ReconOutLoss':'Beta_Rec', 'val_kl_Loss_Z':'Beta_Z' }
    ScalingDic = { 'val_ReconOutLoss':WRec, 'val_kl_Loss_Z':WZ}
    MinLimit = {'Beta_Rec':MnWRec,  'Beta_Z':MnWZ}
    MaxLimit = {'Beta_Rec':MxWRec,  'Beta_Z':MxWZ}
        
        
    #### ------------------------------------------------ Dynamic controller for specific losses and betas ------------------------------------------------
    if 'TCVAE' in ConfigName :
        RelLossDic['val_kl_Loss_TC'] = 'Beta_TC'
        ScalingDic['val_kl_Loss_TC'] = ConfigSpec['WTC']
        MinLimit['Beta_TC'] = ConfigSpec['MnWTC']
        MaxLimit['Beta_TC'] = ConfigSpec['MxWTC']
        
        RelLossDic['val_kl_Loss_MI'] = 'Beta_MI'
        ScalingDic['val_kl_Loss_MI'] = ConfigSpec['WMI']
        MinLimit['Beta_MI'] = ConfigSpec['MnWMI']
        MaxLimit['Beta_MI'] = ConfigSpec['MxWMI']
    
    if 'FACVAE' in ConfigName :
        RelLossDic['val_kl_Loss_TC'] = 'Beta_TC'
        ScalingDic['val_kl_Loss_TC'] = ConfigSpec['WTC']
        MinLimit['Beta_TC'] = ConfigSpec['MnWTC']
        MaxLimit['Beta_TC'] = ConfigSpec['MxWTC']
        
        RelLossDic['val_kl_Loss_DTC'] = 'Beta_DTC'
        ScalingDic['val_kl_Loss_DTC'] = ConfigSpec['WDTC']
        MinLimit['Beta_DTC'] = ConfigSpec['MnWDTC']
        MaxLimit['Beta_DTC'] = ConfigSpec['MxWDTC']
        

    RelLoss = RelLossWeight(BetaList=RelLossDic, LossScaling= ScalingDic, MinLimit= MinLimit, MaxLimit = MaxLimit, SavePath = ModelSaveName, 
                            ToSaveLoss=ToSaveLoss, SaveWay=SaveWay, Buffer=Buffer, Resume=Resume )
    
    return RelLoss