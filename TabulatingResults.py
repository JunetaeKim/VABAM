import os
import pandas as pd
import numpy as np
import re
import pickle
from argparse import ArgumentParser


from Models.Caller import *
from Utilities.Utilities import ReadYaml, SerializeObjects, DeserializeObjects
from BatchMIEvaluation import LoadModelConfigs, LoadParams
from Utilities.EvaluationMain import *
from Utilities.AncillaryFunctions import Denorm, MAPECal, MSECal


# Refer to the execution code
# python .\TabulatingResults.py -CP ./Config/ --GPUID 0
# python .\TabulatingResults.py -CP ./Config/ -NJ 10 -MC 1 -BS 3000  --GPUID 0 EvalConfigART500.yml


def Aggregation (ConfigName, ConfigPath, NJ=1, MetricCut = 1., BatSize=3000):

    print()
    print(ConfigName)
    
    # Configuration and Object part
    print('-----------------------------------------------------' )
    print('Loading configurations and objects' )
    ## Loading the model configurations
    EvalConfigs = ReadYaml(ConfigPath)
    ModelConfigSet, ModelLoadName = LoadModelConfigs(ConfigName)
    
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
    AnalData = np.load('./Data/ProcessedData/Val'+str(Params['SigType'])+'.npy')
    
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
    MIVals.columns = [r'(1) I(V;Z)',r'(2) $I(V; \acute{Z} \mid Z)$',  r'(3) $I(V;\acute{Z})$', r'(4) $I(V;\acute{\Theta} \mid \acute{Z})$', r'(5) $I(S;\acute{Z})$', r'(6) $I(S;\acute{\Theta} \mid \acute{Z})$']
    MIVals['Model'] = ConfigName
    longMI = MIVals.melt(id_vars='Model', var_name='Metrics', value_name='Values')

    return MSEnorm, MSEdenorm, MAPEnorm, MAPEdenorm, longMI, MeanKld_GTTG
    


if __name__ == "__main__":

    
    # Create the parser
    parser = ArgumentParser()
    
    # Add Experiment-related parameters
    parser.add_argument('--ConfigPath', '-CP', type=str, required=True, help='Set the path of the configuration to load (the name of the YAML file).')
    parser.add_argument('-NJ', type=int, required=False, default=1, help='The size of js to be selected at the same time (default: 1).')
    parser.add_argument('--MetricCut', '-MC',type=int, required=False, default=1, help='The threshold for Zs and ancillary data where the metric value is below SelMetricCut (default: 1)')
    parser.add_argument('--BatSize', '-BS',type=int, required=False, default=5000, help='The batch size during prediction.')
    parser.add_argument('--GPUID', type=int, required=False, default=1)
    
    args = parser.parse_args() # Parse the arguments
    YamlPath = args.ConfigPath
    NJ = args.NJ
    MetricCut = args.MetricCut
    BatSize = args.BatSize
    GPU_ID = args.GPUID
   

    ## GPU selection
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "0"
    
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
                 
    Exp = r'ART|II|\d+'  # Regular expression pattern for 'ART' and 'II'.
    
    EvalConfigList = os.listdir(YamlPath) # Retrieve a list of all files in the YamlPath directory.
    EvalConfigList = [i for i in EvalConfigList if 'Eval' in i] # Filter the list to include only files that contain 'Eval' in their names.
    
    # Iterate through each filtered configuration file.
    for EvalConfig in EvalConfigList:
        print(EvalConfig)  
        
        # Read the configuration file's contents.
        ModelConfigs = ReadYaml(YamlPath + EvalConfig) 
        # Filter the configurations to include only those with 'ART' or 'II'.
        ConfigNames = [i for i in ModelConfigs if 'ART' in i or 'II' in i]
    
        # Initialize lists to store results.
        ModelName = []
        MSEnormRes = []
        MSEdenormRes = []
        MAPEnormRes = []
        MAPEdenormRes = []
        MeanKldRes = []
        MItables = pd.DataFrame()  # Initialize an empty DataFrame for MI tables.
        
        # Iterate through each filtered configuration name.
        for ConfigName in ConfigNames:
            # Perform aggregation (custom function) and retrieve results.
            MSEnorm, MSEdenorm, MAPEnorm, MAPEdenorm, longMI, MeanKld_GTTG = Aggregation(ConfigName, YamlPath + EvalConfig, NJ=NJ, MetricCut=MetricCut, BatSize=BatSize)
             
            # Append the results to their respective lists.
            ModelName.append(ConfigName)
            MSEnormRes.append(MSEnorm)
            MSEdenormRes.append(MSEdenorm) 
            MAPEnormRes.append(MAPEnorm)
            MAPEdenormRes.append(MAPEdenorm) # it's reported for reference, it is not used as an official metric in this paper. 
            MeanKldRes.append(MeanKld_GTTG)
            
            # Concatenate the current longMI DataFrame to the MItables DataFrame.
            MItables = pd.concat([MItables, longMI]).copy()
        
        # Extract relevant parts from EvalConfig to name the table.
        TableName = re.findall(Exp, EvalConfig)
        TableName = ''.join(TableName)  # Concatenate the extracted parts.
        
        # Save the MItables to a CSV file.
        MItables.to_csv('./EvalResults/Tables/MI_' + str(TableName) + '.csv', index=False)
    
        # Save the AccKLDtables to a CSV file.
        DicRes = {'Model': ModelName , 'MeanKldRes': MeanKldRes, 'MSEnorm':MSEnormRes , 'MSEdenorm': MSEdenormRes, 'MAPEnorm': MAPEnormRes, 'MAPEdenorm': MAPEdenormRes }
        AccKLDtables = pd.DataFrame(DicRes)
        AccKLDtables.to_csv('./EvalResults/Tables/AccKLD_' + str(TableName) + '_Nj'+str(NJ) +'.csv', index=False)
        
