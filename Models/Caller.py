from Models.MainModel import *
from Models.Losses import *
from Models.Discriminator import FacDiscriminator


def ModelCall (SelConfigSet, SigDim, DataSize, Resume=False, LoadWeight=False, Reparam=True, ModelSaveName=None):
    
    assert not (Resume or LoadWeight) or ModelSaveName is not None, "ModelSaveName must be provided to load the weights."
    
    ### Model-related parameters
    LatDim = SelConfigSet['LatDim']
    CompSize = SelConfigSet['CompSize']
    assert CompSize in [i for i in range(100, 1000, 100)], "Value should be one of " +str([i for i in range(100, 1000, 100)])
    MaskingRate = SelConfigSet['MaskingRate']
    NoiseStd = SelConfigSet['NoiseStd']
    MaskStd = SelConfigSet['MaskStd']
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
        
    elif LossType =='FACLosses':
        DiscHiddenSize = SelConfigSet['DiscHiddenSize']
        FacDiscModel = FacDiscriminator(LatDim, DiscHiddenSize)
        Models = [EncModel,FeatExtModel,FeatGenModel,ReconModel, FacDiscModel] 
        SigRepModel = FACLosses(Models, SelConfigSet)

        
    # Model Compile
    SigRepModel.compile(optimizer='adam') 
    SigRepModel.summary()
    
    
    # Model Training
    if Resume == True or LoadWeight == True:
        SigRepModel.load_weights(ModelSaveName)
        print('Model weights loaded')
        
    return SigRepModel