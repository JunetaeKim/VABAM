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
        #Tr_Cond_3d = FFT_PSD(tf.signal.frame(TrRaw, SlidingSize, SlidingSize), 'None')
        #Val_Cond_3d = FFT_PSD(tf.signal.frame(ValRaw, SlidingSize, SlidingSize), 'None')
        Tr_Cond = FFT_PSD(TrRaw, 'None')[:, 0]
        Val_Cond = FFT_PSD(ValRaw, 'None')[:, 0]      

        TrInp, ValInp = [TrSampled, Tr_Cond], [ValSampled, Val_Cond]
        ConDim = Tr_Cond.shape[-1]

        # Call the model with some dummy input data to create the variables and allows weight loading
        BenchModel = Wavenet(SigDim, ConfigSpec, ConDim, SlidingSize = SlidingSize)
        BenchModel([Dummy[:1] for Dummy in TrInp])
    
    elif 'DiffWave' in ConfigName:
        Tr_Cond = FFT_PSD(TrData, 'None')[:, 0]
        Val_Cond = FFT_PSD(ValData, 'None')[:, 0]   
        TrInp, ValInp = [TrData, Tr_Cond], [ValData, Val_Cond]
        ConfigSpec['SigDim'] = TrData.shape[-1]

        # Call the model with some dummy input data to create the variables and allows weight loading
        BenchModel = DiffWave(ConfigSpec)
        _ = BenchModel.wavenet(TrInp[0][:1], tf.convert_to_tensor([1], dtype=tf.int32), TrInp[1][:1])
        _ = BenchModel(TrInp[1][:1])

    else:
        assert False, "Please verify if the model name is right or not."    

    
    # Model Training
    if Resume == True or LoadWeight == True:
        BenchModel.load_weights(ModelSaveName)
        print('Model weights loaded')
        
    return BenchModel, TrInp, ValInp



# Dynamic controller for losses
def DCLCall(ConfigSpec, ConfigName, ModelSaveName,
            ToSaveLoss=None, SaveWay='max', Buffer=0, Resume=False):
    """
    Dynamic controller for losses.
    
    - 'WRec', 'MnWRec', and 'MxWRec' are mandatory in ConfigSpec.
    - Other parameters ('WZ', 'WTC', 'WDTC', 'WMI', etc.) are optional:
      they are only added if all required parts exist.
    - TCVAE and FACVAE parameters are only added if 'TCVAE' or 'FACVAE'
      is in ConfigName and the corresponding parameters are all found.
    """

    # Default list of losses to save if not specified
    if ToSaveLoss is None:
        ToSaveLoss = ['val_ReconOutLoss']


    # Initialize dictionaries with mandatory Reconstruction weights.
    RelLossDic = {'val_ReconOutLoss': 'Beta_Rec'}
    ScalingDic = {'val_ReconOutLoss': ConfigSpec['WRec']}
    MinLimit = {'Beta_Rec': ConfigSpec['MnWRec']}
    MaxLimit = {'Beta_Rec':  ConfigSpec['MxWRec']}


    # Conditionally add Z-KLD
    WZ = ConfigSpec.get('WZ', None)
    if WZ is not None:
        RelLossDic['val_kl_Loss_Z'] = 'Beta_Z'
        ScalingDic['val_kl_Loss_Z'] = WZ
        MinLimit['Beta_Z'] = ConfigSpec.get('MnWZ', None)
        MaxLimit['Beta_Z'] = ConfigSpec.get('MxWZ', None)

    # Conditionally add TC
    WTC = ConfigSpec.get('WTC', None)
    if WTC is not None:
        RelLossDic['val_kl_Loss_TC'] = 'Beta_TC'
        ScalingDic['val_kl_Loss_TC'] = WTC
        MinLimit['Beta_TC'] = ConfigSpec.get('MnWTC', None)
        MaxLimit['Beta_TC'] = ConfigSpec.get('MxWTC', None)

    # Conditionally add MI
    WMI = ConfigSpec.get('WMI', None)
    if WMI is not None:
        RelLossDic['val_kl_Loss_MI'] = 'Beta_MI'
        ScalingDic['val_kl_Loss_MI'] = WMI
        MinLimit['Beta_MI'] = ConfigSpec.get('MnWMI', None)
        MaxLimit['Beta_MI'] = ConfigSpec.get('MxWMI', None)

    # Conditionally add DTC
    WDTC = ConfigSpec.get('WDTC', None)
    if WDTC is not None:
        RelLossDic['val_kl_Loss_DTC'] = 'Beta_DTC'
        ScalingDic['val_kl_Loss_DTC'] = WDTC
        MinLimit['Beta_DTC'] = ConfigSpec.get('MnWDTC', None)
        MaxLimit['Beta_DTC'] = ConfigSpec.get('MxWDTC', None)
                

    RelLoss = RelLossWeight(BetaList=RelLossDic, LossScaling= ScalingDic, MinLimit= MinLimit, MaxLimit = MaxLimit, 
                            SavePath = ModelSaveName, ToSaveLoss=ToSaveLoss, SaveWay=SaveWay, Buffer=Buffer, Resume=Resume )
    
    return RelLoss