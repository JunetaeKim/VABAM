import sys
# setting path
sys.path.append('../')

import os
import re
import numpy as np
from argparse import ArgumentParser
import pandas as pd
import pickle

from Benchmarks.Models.BenchmarkCaller import *
from Utilities.EvaluationMain import *
from Utilities.Utilities import ReadYaml, SerializeObjects, DeserializeObjects, LoadModelConfigs, LoadParams
from Utilities.AncillaryFunctions import Denorm, MAPECal, MSECal


# Refer to the execution code
# python .\TabulatingBMResults.py -CP ./Config/ --GPUID 0


def Aggregation (ConfigName, ConfigPath, NJ=1, MetricCut = 1., BatSize=3000):

    print()
    print(ConfigName)
    
    # Configuration and Object part
    print('-----------------------------------------------------' )
    print('Loading configurations and objects' )
    ## Loading the model configurations
    EvalConfigs = ReadYaml(ConfigPath)
    ModelConfigSet, ModelLoadName = LoadModelConfigs(ConfigName, Comp=False)

    if ModelConfigSet['LatDim'] < NJ:
        return None, None, None, None, None, None  # To ensure consistency with the main return statement, return 6 None
        
    ## Loading parameters for the evaluation
    Params = LoadParams(ModelConfigSet, EvalConfigs[ConfigName])
    Params['Common_Info'] = EvalConfigs['Common_Info']
    
    ## Object Load path
    ObjLoadPath = './EvalResults/Instances/Obj_'+ConfigName+'_Nj'+str(NJ)+'.pkl'
    SampZjLoadPath = './Data/IntermediateData/'+ConfigName+'_SampZj_'+str(Params['NSelZ'])+'.npy'
    
    
    # Data part
    print('-----------------------------------------------------' )
    print('Loading data')
    ## Loading data
    TestData = np.load('../Data/ProcessedData/Test'+str(Params['SigType'])+'.npy')
    TrData = np.load('../Data/ProcessedData/Tr'+str(Params['SigType'])+'.npy')
    
    
    ## Intermediate parameters 
    SigDim = TestData.shape[1]
    DataSize = TestData.shape[0]
    
    with open('../Data/ProcessedData/SigMax.pkl', 'rb') as f:
        SigMax = pickle.load(f)
    with open('../Data/ProcessedData/SigMin.pkl', 'rb') as f:
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
    BenchModel, _, AnalData = ModelCall (ModelConfigSet, ConfigName, TrData, TestData, LoadWeight=True, Reparam=False, ReparaStd=Params['ReparaStd'], ModelSaveName=ModelLoadName)
    
    
    ## The generation model for evaluation
    GenModel = BenchModel.get_layer('ReconModel')
    
    ## The sampling model for evaluation
    Inp_Enc = BenchModel.get_layer('Inp_Enc')
    Zs = BenchModel.get_layer('Zs').output
    
    if Params['SecDataType'] == 'CONDIN':
        Inp_Cond = BenchModel.get_layer('Inp_Cond')
        SampModel = Model([Inp_Enc.input, Inp_Cond.input], Zs)
    else:
        SampModel = Model(Inp_Enc.input, Zs)
    
    
    # Evaluating MAPEs
    ## Prediction
    print('-----------------------------------------------------' )
    print('MAPE calculation')
    
    PredRes = BenchModel.predict(AnalData, batch_size=BatSize, verbose=1)
    if 'FAC' in ConfigName:
        PredSigRec = PredRes[-1]
    else:
        PredSigRec = PredRes
    
    if Params['SecDataType'] == 'CONDIN':
        InpData = AnalData[0]
    else:
        InpData = AnalData
    
    
    ## MAPE    
    MAPEnorm, MAPEdenorm = MAPECal(InpData, PredSigRec, MaxX, MinX)
    ## MSE    
    MSEnorm, MSEdenorm = MSECal(InpData, PredSigRec, MaxX, MinX)
    
    # Evaluating Mutual information
    ## Creating new instances
    NewEval = Evaluator()
    # Populating it with the saved data
    DeserializeObjects(NewEval, ObjLoadPath)
    if Params['SecDataType'] == 'CONDIN':
        NewEval.SecDataType = 'CONDIN'
    else:
        NewEval.SecDataType = False
    
    # Post evaluation of KLD
    ## MetricCut: The threshold value for selecting Zs whose Entropy of PSD (i.e., SumH) is less than the MetricCut
    PostSamp = NewEval.SelPostSamp( MetricCut)

    ## Calculation of KLD
    NewEval.GenModel = GenModel
    NewEval.KLD_TrueGen(AnalSig=InpData, PlotDist=False) 
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
    if Params['SecDataType'] == 'CONDIN':
        MIVals.columns = [r'(i) I(V;Z)',r'(ii) $I(V; \acute{Z} \mid Z)$',  r'(iii) $I(V;\acute{Z})$', r'(iv) $I(V;\acute{\Theta} \mid \acute{Z})$', r'(v) $I(S;\acute{Z})$', r'(vi) $I(S;\acute{\Theta} \mid \acute{Z})$']
    else:
        MIVals.columns = [r'(i) I(V;Z)',r'(ii) $I(V; \acute{Z} \mid Z)$']
        
    MIVals['Model'] = ConfigName
    longMI = MIVals.melt(id_vars='Model', var_name='Metrics', value_name='Values')

    return MSEnorm, MSEdenorm, MAPEnorm, MAPEdenorm, longMI, MeanKld_GTTG
    
# Function to extract Nj value from filename
def ExtractNj(Filename):
    Match = re.search(r'Nj(\d+)\.', Filename)
    return int(Match.group(1)) if Match else None

if __name__ == "__main__":

    
    # Create the parser
    parser = ArgumentParser()
    
    # Add Experiment-related parameters
    parser.add_argument('--ConfigPath', '-CP', type=str, required=True, help='Set the path of the configuration to load (the name of the YAML file).')
    parser.add_argument('--MetricCut', '-MC',type=int, required=False, default=1, help='The threshold for Zs and ancillary data where the metric value is below SelMetricCut (default: 1)')
    parser.add_argument('--BatSize', '-BS',type=int, required=False, default=5000, help='The batch size during prediction.')
    parser.add_argument('--GPUID', type=int, required=False, default=1)
    
    args = parser.parse_args() # Parse the arguments
    YamlPath = args.ConfigPath
    MetricCut = args.MetricCut
    BatSize = args.BatSize
    GPU_ID = args.GPUID
   

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
    
    ## Loading the model configuration lists
    EvalConfigList = os.listdir(YamlPath) # Retrieve a list of all files in the YamlPath directory.
    EvalConfigList = [i for i in EvalConfigList if 'Eval' in i] # Filter the list to include only files that contain 'Eval' in their names.

       
    # loop
    for Filename in FileList:
        # Extracts the string between 'Obj_' and '_Nj'
        ConfigName =re.search(r'Obj_(.*?)_Nj', Filename).group(1)  
        ConfigPath = [EvalConfig for EvalConfig in EvalConfigList if ConfigName.split('_')[1] in EvalConfig ][0]
        ConfigPath = YamlPath + ConfigPath
        NJ = ExtractNj(Filename)
        # Perform aggregation (custom function) and retrieve results.
        MSEnorm, MSEdenorm, MAPEnorm, MAPEdenorm, longMI, MeanKld_GTTG = Aggregation(ConfigName, ConfigPath, NJ=NJ, MetricCut=MetricCut, BatSize=BatSize)
    
        # Save the MItables to a CSV file.
        longMI.to_csv('./EvalResults/Tables/MI_' + str(ConfigName) +'_Nj'+str(NJ)+'.csv', index=False)
    
        # Save the AccKLDtables to a CSV file.
        DicRes = {'Model': [ConfigName] , 'MeanKldRes': [MeanKld_GTTG], 'MSEnorm':[MSEnorm] , 'MSEdenorm': [MSEdenorm], 'MAPEnorm': [MAPEnorm], 'MAPEdenorm': [MAPEdenorm] }
        AccKLDtables = pd.DataFrame(DicRes)
        AccKLDtables.to_csv('./EvalResults/Tables/AccKLD_' + str(ConfigName) + '_Nj'+str(NJ)+'.csv', index=False)
        
