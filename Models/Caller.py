from Models.MainModel import *
from Models.Losses import *
from Models.Discriminator import FacDiscriminator
from Utilities.Utilities import RelLossWeight


def ModelCall (SelConfigSet, SigDim, DataSize, Resume=False, LoadWeight=False, ReturnModelPart=False, Reparam=True, ReparaStd=None, ModelSaveName=None):
    
    assert not (Resume or LoadWeight) or ModelSaveName is not None, "ModelSaveName must be provided to load the weights."
    
    ### Model-related parameters
    LatDim = SelConfigSet['LatDim']
    CompSize = SelConfigSet['CompSize']
    assert CompSize in [i for i in range(100, 1000, 100)], "Value should be one of " +str([i for i in range(100, 1000, 100)])
    MaskingRate = SelConfigSet['MaskingRate']
    NoiseStd = SelConfigSet['NoiseStd']
    MaskStd = SelConfigSet['MaskStd']
    if ReparaStd is None:
        ReparaStd = SelConfigSet['ReparaStd']
    FcLimit = SelConfigSet['FcLimit']
    DecayH = SelConfigSet['DecayH']
    DecayL = SelConfigSet['DecayL']
    
    ### Loss-related parameters
    LossType = SelConfigSet['LossType']
           
        
        
    # Defining Modesl
    EncModel = Encoder(SigDim=SigDim, LatDim= LatDim, Type = '', MaskingRate = MaskingRate, NoiseStd = NoiseStd, MaskStd = MaskStd, ReparaStd = ReparaStd, Reparam=Reparam, FcLimit=FcLimit)
    FeatExtModel = FeatExtractor(SigDim=SigDim, CompSize = CompSize, DecayH=DecayH, DecayL=DecayL)
    FeatGenModel = FeatGenerator(SigDim=SigDim,CompSize= CompSize, LatDim= LatDim)
    ReconModel = Reconstructor(SigDim=SigDim, CompSize= CompSize)

    
    # Adding losses
    if LossType =='TCLosses':
        Models = [EncModel,FeatExtModel,FeatGenModel,ReconModel] 
        SigRepModel = TCLosses(Models, DataSize, SelConfigSet)
        ModelParts = [EncModel, FeatExtModel, FeatGenModel, ReconModel]
        
    elif LossType =='FACLosses':
        DiscHiddenSize = SelConfigSet['DiscHiddenSize']
        FacDiscModel = FacDiscriminator(LatDim, DiscHiddenSize)
        Models = [EncModel,FeatExtModel,FeatGenModel,ReconModel, FacDiscModel] 
        SigRepModel = FACLosses(Models, SelConfigSet)
        ModelParts = [EncModel, FeatExtModel, FeatGenModel, ReconModel, FacDiscModel]

        
    # Model Compile
    SigRepModel.compile(optimizer='adam') 
    SigRepModel.summary()
    
    
    # Model Training
    if Resume == True or LoadWeight == True:
        SigRepModel.load_weights(ModelSaveName)
        print('Model weights loaded')
    
    
    if ReturnModelPart == False:
        return SigRepModel
    else:
        return SigRepModel, ModelParts


# Dynamic controller for losses
def DCLCall (SelConfigSet, ModelSaveName, ToSaveLoss=None, SaveWay='max', Resume=False):
    
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
        

    RelLoss = RelLossWeight(BetaList=RelLossDic, LossScaling= ScalingDic, MinLimit= MinLimit, MaxLimit = MaxLimit, SavePath = ModelSaveName, 
                            ToSaveLoss=ToSaveLoss, SaveWay=SaveWay, Resume=Resume)
    
    return RelLoss