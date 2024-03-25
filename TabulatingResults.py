import os
import pandas as pd
import numpy as np
import re
import pickle
from argparse import ArgumentParser
from itertools import product


from Models.Caller import *
from Utilities.Utilities import ReadYaml, SerializeObjects, DeserializeObjects
from BatchMIEvaluation import LoadModelConfigs, LoadParams
from Utilities.EvaluationMain import *
from Utilities.AncillaryFunctions import Denorm, MAPECal, MSECal


# Refer to the execution code
# python .\TabulatingResults.py -CP ./Config/ --GPUID 0


def Aggregation (ConfigName, ConfigPath, NJ=1, FC=1.0, MetricCut = 1., BatSize=3000):

    print()
    print(ConfigName)
    
    # Configuration and Object part
    print('-----------------------------------------------------' )
    print('Loading configurations and objects' )
    ## Loading the model configurations
    EvalConfigs = ReadYaml(ConfigPath)
    ModelConfigSet, ModelLoadName = LoadModelConfigs(ConfigName)

    if ModelConfigSet['LatDim'] < NJ:
        return None, None, None, None, None, None  # To ensure consistency with the main return statement, return 6 None
    
    ## Loading parameters for the evaluation
    Params = LoadParams(ModelConfigSet, EvalConfigs[ConfigName])
    Params['Common_Info'] = EvalConfigs['Common_Info']
    
    ## Object Load path
    ObjLoadPath = './EvalResults/Instances/Obj_'+ConfigName+'_Nj'+str(NJ)+'_FC'+str(FC)+'.pkl'


    # Data part
    print('-----------------------------------------------------' )
    print('Loading data')
    ## Loading data
    AnalData = np.load('./Data/ProcessedData/Test'+str(Params['SigType'])+'.npy')
    
    ## Intermediate parameters 
    SigDim = AnalData.shape[1]
    DataSize = AnalData.shape[0]
    
    with open('./Data/ProcessedData/SigMax.pkl', 'rb') as f:
        SigMax = pickle.load(f)
    with open('./Data/ProcessedData/SigMin.pkl', 'rb') as f:
        SigMin = pickle.load(f)
    
    
    if 'ART' in ConfigName:
        MaxX, MinX = SigMax['ART'], SigMin['ART']
    elif 'PLETH' in ConfigName:
        MaxX, MinX = SigMax['PLETH'], SigMin['PLETH']
    elif 'II' in ConfigName:
        MaxX, MinX = SigMax['II'], SigMin['II']


    
    # Model part
    print('-----------------------------------------------------' )
    print('Loading model structures')
    ## Calling Modesl
    SigRepModel, ModelParts = ModelCall (ModelConfigSet, SigDim, DataSize, LoadWeight=True, ReturnModelPart=True, Reparam=False, ReparaStd=Params['ReparaStd'], ModelSaveName=ModelLoadName, ModelSummary=False)
    
    ## Setting Model Specifications and Sub-models
    if Params['LossType'] =='Default':
        EncModel, FeatExtModel, FeatGenModel, ReconModel = ModelParts
    elif Params['LossType'] =='FACLosses':
        EncModel, FeatExtModel, FeatGenModel, ReconModel, FacDiscModel = ModelParts
            
    ## The generation model for evaluation
    RecOut = ReconModel(FeatGenModel.output)
    GenModel = Model(FeatGenModel.input, RecOut)

    ## The sampling model for evaluation
    Zs_Out = SigRepModel.get_layer('Zs').output
    SampModel = Model(EncModel.input, Zs_Out)


    # Evaluating MAPEs
    ## Prediction
    print('-----------------------------------------------------' )
    print('MAPE calculation')
    PredSigRec = SigRepModel.predict(AnalData, batch_size=BatSize, verbose=1)[-2]
   
    ## MAPE    
    MAPEnorm, MAPEdenorm = MAPECal(AnalData, PredSigRec, MaxX, MinX)
    ## MSE    
    MSEnorm, MSEdenorm = MSECal(AnalData, PredSigRec, MaxX, MinX)

    # Evaluating Mutual information
    ## Creating new instances
    NewEval = Evaluator()
    # Populating it with the saved data
    DeserializeObjects(NewEval, ObjLoadPath)

    # Post evaluation of KLD
    ## MetricCut: The threshold value for selecting Zs whose Entropy of PSD (i.e., SumH) is less than the MetricCut
    PostSamp = NewEval.SelPostSamp( MetricCut)

    ## Calculation of KLD
    NewEval.GenModel = GenModel
    NewEval.KLD_TrueGen(AnalSig=AnalData, PlotDist=False) 
    MeanKld_GTTG = (NewEval.KldPSD_GenTrue + NewEval.KldPSD_TrueGen) / 2

    print(MeanKld_GTTG)

    ''' Renaming columns '''
    # r'I(V;Z)'
    # r'I(V; \acute{Z} \mid Z)'
    # r'I(V;\acute{Z})'
    # r'I(V;\acute{\Theta} \mid \acute{Z})'
    # r'I(S;\acute{Z})'
    # r'I(S;\acute{\Theta} \mid \acute{Z})'
    
    MIVals = pd.DataFrame(NewEval.SubResDic)
    MIVals.columns = [r'(i) I(V;Z)',r'(ii) $I(V; \acute{Z} \mid Z)$',  r'(iii) $I(V;\acute{Z})$', r'(iv) $I(V;\acute{\Theta} \mid \acute{Z})$', r'(v) $I(S;\acute{Z})$', r'(vi) $I(S;\acute{\Theta} \mid \acute{Z})$']
    MIVals['Model'] = ConfigName
    longMI = MIVals.melt(id_vars='Model', var_name='Metrics', value_name='Values')

    return MSEnorm, MSEdenorm, MAPEnorm, MAPEdenorm, longMI, MeanKld_GTTG
    


def ExtractFC(Filename):
    Match = re.search(r'FC(\d+\.?\d*)', Filename)
    return Match.group(1) if Match else None

# Function to extract Nj value from filename
def ExtractNj(Filename):
    Match = re.search(r'Nj(\d+)\_', Filename)
    return int(Match.group(1)) if Match else None



if __name__ == "__main__":

    
    # Create the parser
    parser = ArgumentParser()
    
    # Add Experiment-related parameters
    parser.add_argument('--ConfigPath', '-CP', type=str, required=True, help='Set the path of the configuration to load (the name of the YAML file).')
    parser.add_argument('--ConfigSpec', nargs='+', type=str, required=False, default=None, help='Set the name of the specific configuration to load (the name of the model config in the YAML file).')
    parser.add_argument('--MetricCut', '-MC',type=int, required=False, default=1, help='The threshold for Zs and ancillary data where the metric value is below SelMetricCut (default: 1)')
    parser.add_argument('--BatSize', '-BS',type=int, required=False, default=5000, help='The batch size during prediction.')
    parser.add_argument('--GPUID', type=int, required=False, default=1)
    parser.add_argument('--SpecNZs', '-NJ', nargs='+', type=int, required=False, default=None, help='Set the size of js to be selected at the same time with the list.')
    parser.add_argument('--SpecFCs', nargs='+', type=float, required=False, default=None, help='Set the frequency cutoff range(s) for signal synthesis. Multiple ranges can be provided.')

    args = parser.parse_args() # Parse the arguments
    YamlPath = args.ConfigPath
    MetricCut = args.MetricCut
    BatSize = args.BatSize
    GPU_ID = args.GPUID
    SpecNZs = args.SpecNZs
    SpecFCs = args.SpecFCs
    ConfigSpec = args.ConfigSpec

  

    ## GPU selection
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_ID)
    
    # TensorFlow memory configuration
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            gpu = gpus[0]  # Fix the index as zero since GPU_ID has already been given. 
            tf.config.experimental.set_memory_growth(gpu, False)
            tf.config.experimental.set_virtual_device_configuration
            (
                gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024*23.5))]  
            )
        except RuntimeError as e:
            print(e)            
    
                 
                 
                 
    #### -----------------------------------------------------  Conducting tabulation --------------------------------------------------------------
                 
    # Object part
    print('-----------------------------------------------------' )
    print('Scanning objects' )
    print('-----------------------------------------------------' )
    ObjLoadPath = './EvalResults/Instances/'
    FileList = os.listdir(ObjLoadPath)
    FileList = [file for file in FileList if file.endswith('.pkl')]

    
    ## Loading the model configuration lists
    EvalConfigList = os.listdir(YamlPath) # Retrieve a list of all files in the YamlPath directory.
    EvalConfigList = [i for i in EvalConfigList if 'Eval' in i] # Filter the list to include only files that contain 'Eval' in their names.

    
    # Filter the files to include only those in ConfigSpec
    if ConfigSpec is not None:
        FileList = [Filename for Filename in FileList if any(Config in Filename for Config in ConfigSpec) ]
    
    # Filter the files to include only those with FC values in SpecFCs
    if SpecFCs is not None:
        FileList = [Filename for Filename in FileList if any(f'FC{fc}' in Filename for fc in SpecFCs)]
    
    # Filter the files to include only those with Nz in SpecNZs
    if SpecNZs is not None:
        FileList = [Filename for Filename in FileList if any(f'Nj{Nj}' in Filename for Nj in SpecNZs)]

    print(FileList)
    
    
    # loop
    for Filename in FileList:
        # Extracts the string between 'Obj_' and '_Nj'
        ConfigName =re.search(r'Obj_(.*?)_Nj', Filename).group(1)  
        ConfigPath = [EvalConfig for EvalConfig in EvalConfigList if ConfigName.split('_')[1] in EvalConfig and ConfigName.split('_')[-1] in EvalConfig][0]
        ConfigPath = YamlPath + ConfigPath
        NJ = ExtractNj(Filename)
        FC = ExtractFC(Filename)
        # Perform aggregation (custom function) and retrieve results.
        MSEnorm, MSEdenorm, MAPEnorm, MAPEdenorm, longMI, MeanKld_GTTG = Aggregation(ConfigName, ConfigPath, NJ=NJ, FC=FC, MetricCut=MetricCut, BatSize=BatSize)
    
        # Save the MItables to a CSV file.
        longMI.to_csv('./EvalResults/Tables/MI_' + str(ConfigName) +'_Nj'+str(NJ)+'_FC'+str(FC) + '.csv', index=False)
    
        # Save the AccKLDtables to a CSV file.
        DicRes = {'Model': [ConfigName] , 'MeanKldRes': [MeanKld_GTTG], 'MSEnorm':[MSEnorm] , 'MSEdenorm': [MSEdenorm], 'MAPEnorm': [MAPEnorm], 'MAPEdenorm': [MAPEdenorm] }
        AccKLDtables = pd.DataFrame(DicRes)
        AccKLDtables.to_csv('./EvalResults/Tables/AccKLD_' + str(ConfigName) + '_Nj'+str(NJ)+'_FC'+str(FC) +'.csv', index=False)
            
