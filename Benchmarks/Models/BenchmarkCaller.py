from Benchmarks.Models.BaseModels import *
from Benchmarks.Models.BenchmarkModels import *
from Utilities.Utilities import RelLossWeight


def ModelCall (ConfigSpec, ConfigName, TrData, ValData, Resume=False, LoadWeight=False, Reparam=True, ModelSaveName=None):
    
    assert not (Resume or LoadWeight) or ModelSaveName is not None, "ModelSaveName must be provided to load the weights."
    DataSize = TrData.shape[0] 
    SigDim = TrData.shape[1]
    
        
    # ModelName selection
    if 'BaseVAE' in ConfigName:
        BenchModel = BaseVAE(SigDim, ConfigSpec, Reparam=Reparam)
        TrInp, ValInp = TrData, ValData
    
    elif 'TCVAE' in ConfigName:
        BenchModel = TCVAE(SigDim, DataSize, ConfigSpec, Reparam=Reparam)
        TrInp, ValInp = TrData, ValData
    
    elif 'FACVAE' in ConfigName:
        BenchModel = FACVAE(SigDim, ConfigSpec, Reparam=Reparam)
        TrInp, ValInp = TrData, ValData
    
    elif 'ConVAE' in ConfigName:
        ## Identifying conditions based on cumulative Power Spectral Entropy (PSE) over each frequency
        Tr_Cond = FFT_PSD(TrData, 'None')[:, 0]
        Val_Cond = FFT_PSD(ValData, 'None')[:, 0]
        TrInp, ValInp = [TrData, Tr_Cond], [ValData, Val_Cond]
    
        CondDim = Tr_Cond.shape[-1]
        BenchModel= ConVAE(SigDim, CondDim, ConfigSpec, Reparam=Reparam)
         
    else:
        assert False, "Please verify if the model name is right or not."    

    
    # Model Training
    if Resume == True or LoadWeight == True:
        BenchModel.load_weights(ModelSaveName)
        print('Model weights loaded')
        
    return BenchModel, TrInp, ValInp



# Dynamic controller for losses
def DCLCall (SelConfigSet, ModelSaveName, ToSaveLoss=None, SaveWay='max'):
    
    if ToSaveLoss is None:
        ToSaveLoss = ['val_FeatRecLoss', 'val_OrigRecLoss']
    
    ### Loss-related parameters
    LossType = SelConfigSet['LossType']
    SpecLosses = SelConfigSet['SpecLosses']
    
    ### Parameters for constant losse weights
    WRec = SelConfigSet['WRec']
    WFeat = SelConfigSet['WFeat']
    WZ = SelConfigSet['WZ']
    
    ### Parameters for dynamic controller for losse weights
    MnWRec = SelConfigSet['MnWRec']
    MnWFeat = SelConfigSet['MnWFeat']
    MnWZ = SelConfigSet['MnWZ']
    
    MxWRec = SelConfigSet['MxWRec']
    MxWFeat = SelConfigSet['MxWFeat']
    MxWZ = SelConfigSet['MxWZ']
    
    
    ### Dynamic controller for common losses and betas; The relative size of the loss is reflected in the weight to minimize the loss.
    RelLossDic = { 'val_OrigRecLoss':'Beta_Orig', 'val_FeatRecLoss':'Beta_Feat', 'val_kl_Loss_Z':'Beta_Z'}
    ScalingDic = { 'val_OrigRecLoss':WRec, 'val_FeatRecLoss':WFeat, 'val_kl_Loss_Z':WZ}
    MinLimit = {'Beta_Orig':MnWRec, 'Beta_Feat':MnWFeat, 'Beta_Z':MnWZ}
    MaxLimit = {'Beta_Orig':MxWRec, 'Beta_Feat':MxWFeat, 'Beta_Z':MxWZ}
    
    
    ### Dynamic controller for specific losses and betas
    if 'FC' in SpecLosses :
        RelLossDic['val_kl_Loss_FC'] = 'Beta_Fc'
        ScalingDic['val_kl_Loss_FC'] = SelConfigSet['WFC']
        MinLimit['Beta_Fc'] = SelConfigSet['MnWFC']
        MaxLimit['Beta_Fc'] = SelConfigSet['MxWFC']
        
    if 'TC' in SpecLosses :
        RelLossDic['val_kl_Loss_TC'] = 'Beta_TC'
        ScalingDic['val_kl_Loss_TC'] = SelConfigSet['WTC']
        MinLimit['Beta_TC'] = SelConfigSet['MnWTC']
        MaxLimit['Beta_TC'] = SelConfigSet['MxWTC']
        
    if 'MI' in SpecLosses :
        RelLossDic['val_kl_Loss_MI'] = 'Beta_MI'
        ScalingDic['val_kl_Loss_MI'] = SelConfigSet['WMI']
        MinLimit['Beta_MI'] = SelConfigSet['MnWMI']
        MaxLimit['Beta_MI'] = SelConfigSet['MxWMI']
        
    if LossType =='FACLosses':
        RelLossDic['val_kl_Loss_TC'] = 'Beta_TC'
        ScalingDic['val_kl_Loss_TC'] = SelConfigSet['WTC']
        MinLimit['Beta_TC'] = SelConfigSet['MnWTC']
        MaxLimit['Beta_TC'] = SelConfigSet['MxWTC']
        
        RelLossDic['val_kl_Loss_DTC'] = 'Beta_DTC'
        ScalingDic['val_kl_Loss_DTC'] = SelConfigSet['WDTC']
        MinLimit['Beta_DTC'] = SelConfigSet['MnWDTC']
        MaxLimit['Beta_DTC'] = SelConfigSet['MxWDTC']
        

    RelLoss = RelLossWeight(BetaList=RelLossDic, LossScaling= ScalingDic, MinLimit= MinLimit, MaxLimit = MaxLimit, SavePath = ModelSaveName, ToSaveLoss=ToSaveLoss , SaveWay=SaveWay )
    
    return RelLoss