import os
import re
import numpy as np
import matplotlib.pyplot as plt
import copy
from argparse import ArgumentParser

from Models.Caller import *
from Utilities.EvaluationMain import *
from Utilities.Utilities import ReadYaml, SerializeObjects, DeserializeObjects, LoadModelConfigs, LoadParams


# Refer to the execution code
# python .\BatchMIEvaluation.py --Config EvalConfigART800 --GPUID 0
# python .\BatchMIEvaluation.py --Config EvalConfigART800 --ConfigSpec FACFC_ART_50_800  --GPUID 0
# python .\BatchMIEvaluation.py --Config EvalConfigART800 --ConfigSpec FACFC_ART_50_800  --GPUID 0 --SpecNZs 40 50

    
if __name__ == "__main__":

    
    # Create the parser
    parser = ArgumentParser()
    
    # Add Experiment-related parameters
    parser.add_argument('--Config', type=str, required=True, help='Set the name of the configuration to load (the name of the YAML file).')
    parser.add_argument('--ConfigSpec', nargs='+', type=str, required=False, 
                        default=None, help='Set the name of the specific configuration to load (the name of the model config in the YAML file).')
    parser.add_argument('--SpecNZs', nargs='+', type=int, required=False, 
                        default=None, help='Set the size of js to be selected at the same time with the list.')
    parser.add_argument('--GPUID', type=int, required=False, default=1)
    
    args = parser.parse_args() # Parse the arguments
    ConfigName = args.Config
    ConfigSpecName = args.ConfigSpec
    SpecNZs = args.SpecNZs
    GPU_ID = args.GPUID
    
    YamlPath = './Config/'+ConfigName+'.yml'
    EvalConfigs = ReadYaml(YamlPath)

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

    
    # Checking whether the path to save the object exists or not.
    if not os.path.exists('./EvalResults/Instances/') :
        os.makedirs('./EvalResults/Instances/')
                 
    # Checking whether the path to save the SampZj exists or not.
    if not os.path.exists('./Data/IntermediateData/') :
        os.makedirs('./Data/IntermediateData/')
    
                 
                 
                 
    #### -----------------------------------------------------  Conducting batch evaluation --------------------------------------------------------------
                 
    SigTypePrev = None
    for ConfigName in EvalConfigs:
        
        if ConfigName == 'Common_Info':
            continue
        
        if ConfigSpecName is not None: 
            if ConfigName not in ConfigSpecName:
                continue
                
        print()
        print(ConfigName)
        print()
                
        #### -----------------------------------------------------  Setting evaluation environment ----------------------------------------------------------
        # Loading the model configurations
        ModelConfigSet, ModelLoadPath = LoadModelConfigs(ConfigName, Training=False)

        # Loading parameters for the evaluation
        Params = LoadParams(ModelConfigSet, EvalConfigs[ConfigName])
        Params['Common_Info'] = EvalConfigs['Common_Info']


        #### -----------------------------------------------------   Loading data -------------------------------------------------------------------------   
        if SigTypePrev != Params['SigType']:
            SigTypePrev = Params['SigType'] # To change data type: ART, II, PLETH

            # Loading data
            AnalData = np.load('./Data/ProcessedData/Val'+str(Params['SigType'])+'.npy')
            AnalData = np.random.permutation(AnalData)[:Params['EvalDataSize']]

        # Intermediate parameters 
        SigDim = AnalData.shape[1]
        DataSize = AnalData.shape[0]



        #### -----------------------------------------------------   Defining model structure -----------------------------------------------------------------    
        # Calling Modesl
        SigRepModel, ModelParts = ModelCall (ModelConfigSet, SigDim, DataSize, LoadWeight=True, ReturnModelPart=True, ReparaStd=Params['ReparaStd'], ModelSaveName=ModelLoadPath)

        # Intermediate parameters 
        NFCs = SigRepModel.get_layer('FCs').output.shape[-1]


        # Setting Model Specifications and Sub-models
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



        #### -----------------------------------------------------  Conducting Evalution -----------------------------------------------------------------          # Is the value assigned by ArgumentParser or assigned by YML?
        if SpecNZs == None:
            NSelZs = Params['NSelZ']
        else:
            NSelZs = SpecNZs

        for NZs in NSelZs:

            # Object save path
            ObjSavePath = './EvalResults/Instances/Obj_'+ConfigName+'_Nj'+str(NZs)+'.pkl'
            SampZjSavePath = './Data/IntermediateData/'+ConfigName+'_Nj'+str(NZs)+'.npy'
        
            # Instantiation 
            Eval = Evaluator(MinFreq = Params['MinFreq'], MaxFreq = Params['MaxFreq'], SimSize = Params['SimSize'], NMiniBat = Params['NMiniBat'], NParts = Params['NParts'],
                   NGen = Params['NGen'], ReparaStdZj = Params['ReparaStdZj'], NSelZ = NZs, SampBatchSize = Params['SampBatchSize'], 
                   SelMetricType = Params['SelMetricType'], SelMetricCut = Params['SelMetricCut'], GenBatchSize = Params['GenBatchSize'], GPU = Params['GPU'], Name=ConfigName+'_Nj'+str(NZs))
    
            
            ## SampZType: Z~ N(Zμ|y, σ) (SampZType = 'ModelRptA' or 'ModelRptB') vs. Z ~ N(0, ReparaStdZj) (SampZType = 'Gauss' or 'GaussRptA')
            Eval.Eval_ZFC(AnalData[:], SampModel, GenModel, FcLimit=Params['FcLimit'], WindowSize=Params['WindowSize'], Continue=False)
    
    
            # Selecting post Samp_Zj for generating plausible signals
            SelPostSamp = Eval.SelPostSamp( Params['SelMetricCut'], SavePath=SampZjSavePath)
    
    
            # Evaluating KLD (P || K)
            #Eval.KLD_TrueGen(SecDataType ='FCA', RepeatSize = 1, PlotDist=False) 
    
            # Saving the instance's objects to a file
            SerializeObjects(Eval, Params['Common_Info'], ObjSavePath)

            del Eval
    
    
