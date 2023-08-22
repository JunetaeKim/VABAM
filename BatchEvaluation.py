import os
import numpy as np
import re
import matplotlib.pyplot as plt
import copy
from argparse import ArgumentParser

from Models.Caller import *
from Utilities.EvaluationMain import *
from Utilities.Utilities import ReadYaml, SerializeObjects, DeserializeObjects


def LoadModelConfigs(ConfigName):
    
    CompSize = re.findall(r'\d+', ConfigName)[-1]

    if 'ART' in ConfigName:
        LoadConfig = 'Config' + 'ART' + CompSize
        SubPath = 'ART/'
    elif 'PLETH' in ConfigName:
        LoadConfig = 'Config' + 'PLETH' + CompSize
        SubPath = 'PLETH/'
    elif 'II' in ConfigName:
        LoadConfig = 'Config' + 'II' + CompSize
        SubPath = 'II/'
    else:
        assert False, "Please verify if the data type is properly included in the name of the configuration. The configuration name should be structured as 'Config' + 'data type', such as ConfigART."

    YamlPath = './Config/'+LoadConfig+'.yml'
    ConfigSet = ReadYaml(YamlPath)
    
    
    ### Model load path
    ModelName = ConfigName+'.hdf5'
    ModelLoadName = './Results/'+SubPath+ModelName
    
    return ConfigSet[ConfigName], ModelLoadName



def LoadParams (ModelConfigSet, EvalConfigSet): # Experiment setting

    Params = {}
    
    ### Model-related parameters
    Params['SigType']  = ModelConfigSet['SigType']
    Params['ReparaStd'] = EvalConfigSet['ReparaStd']          # The standard deviation value for Gaussian noise generation used in the reparametrization trick.
    Params['ReparaStdZj'] = EvalConfigSet['ReparaStdZj']      # The standard deviation when sampling Zj (Samp_ZjRPT ~ N(0, ReparaStdZj)).
    Params['FcLimit'] = EvalConfigSet['FcLimit']              # The threshold value of the max of the FC value input into the generation model.

    ### Loss-related parameters
    Params['LossType'] = ModelConfigSet['LossType']
    Params['SpecLosses'] = ModelConfigSet['SpecLosses']
    
    ### Evaluation-related parameters
    Params['MaxFreq'] = EvalConfigSet['MaxFreq']              # The maximum frequency value within the analysis range (default = 51).
    Params['MinFreq'] = EvalConfigSet['MinFreq']              # The minimum frequency value within the analysis range (default = 1).
    Params['MinFreqR'] = EvalConfigSet['MinFreqR']            # The minimum value when generating FC_ArangeInp with linspace.
    Params['MaxFreqR'] = EvalConfigSet['MaxFreqR']            # The maximum value when generating FC_ArangeInp with linspace.
    Params['NMiniBat'] = EvalConfigSet['NMiniBat']            # The size of the mini-batch, splitting the task into N pieces of size NMiniBat.
    Params['SimSize'] = EvalConfigSet['SimSize']              # The number of generations (i.e., samplings) within the mini-batch.
    Params['NGen'] = EvalConfigSet['NGen']                    # The number of generations (i.e., samplings) within the mini-batch.
    Params['NSelZ'] = EvalConfigSet['NSelZ']                  # The size of js to be selected at the same time (default: 1).
    Params['MetricCut'] = EvalConfigSet['MetricCut']          # MetricCut: The threshold value for selecting Zs whose Entropy of PSD is less than the MetricCut.
    Params['WindowSize'] = EvalConfigSet['WindowSize']        # The window size when calculating permutation entropy (default: 3)
    Params['SampZType'] = EvalConfigSet['SampZType']          # SampZType: Z~ N(Zμ|y, σ) (SampZType = 'ModelRptA' or 'ModelRptB') vs. 
                                                              # Z ~ N(0, ReparaStdZj) (SampZType = 'Gauss' or 'GaussRptA')
        
    ### Functional parameters
    Params['SampBatchSize'] = EvalConfigSet['SampBatchSize']  # The batch size during prediction of the sampling model.
    Params['GenBatchSize'] = EvalConfigSet['GenBatchSize']    # The batch size during prediction of the generation model.
    Params['GPU'] = EvalConfigSet['GPU']                      # GPU vs CPU during model predictions (i.e., for SampModel and GenModel).
    Params['Common_Info'] = EvalConfigSet['Common_Info']      # The list of common objects subject to class serialization.
    Params['SpecZFC_Info'] = EvalConfigSet['SpecZFC_Info']    # The list of specific objects subject to class serialization.
    
    
    return Params

    

if __name__ == "__main__":

    
    # Create the parser
    parser = ArgumentParser()
    
    # Add Experiment-related parameters
    parser.add_argument('--Config', type=str, required=True, help='Set the name of the configuration to load (the name of the config in the YAML file).')
    parser.add_argument('--GPUID', type=int, required=False, default=1)
    
    args = parser.parse_args() # Parse the arguments
    ConfigName = args.Config
    GPU_ID = args.GPUID
    
    YamlPath = './Config/'+ConfigName+'.yml'
    EvalConfigs = ReadYaml(YamlPath)

    ## GPU selection
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_ID)

    # TensorFlow wizardry
    config = tf.compat.v1.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    # Create a session with the above options specified.
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))         



    
    SigTypePrev = None
    for ConfigName in EvalConfigs:

        #### -----------------------------------------------------  Setting evaluation environment ----------------------------------------------------------
        # Loading the model configurations
        ModelConfigSet, ModelLoadName = LoadModelConfigs(ConfigName)

        # Loading parameters for the evaluation
        Params = LoadParams(ModelConfigSet, EvalConfigs[ConfigName])

        # Object save path
        ObjSavePath = './EvalResults/Instances/Obj_'+ConfigName+'.pkl'
        SampZjSavePath = './Data/IntermediateData/'+ConfigName+'_SampZj_'+str(Params['NSelZ'])+'.npy'


        #### -----------------------------------------------------   Loading data -------------------------------------------------------------------------   
        if SigTypePrev != Params['SigType']:
            SigTypePrev = Params['SigType'] # To change data type: ART, II, PLETH

            # Loading data
            AnalData = np.load('./Data/ProcessedData/Val'+str(Params['SigType'])+'.npy')

        # Intermediate parameters 
        SigDim = AnalData.shape[1]
        DataSize = AnalData.shape[0]



        #### -----------------------------------------------------   Defining model structure -----------------------------------------------------------------    
        # Calling Modesl
        SigRepModel, ModelParts = ModelCall (ModelConfigSet, SigDim, DataSize, LoadWeight=True, ReturnModelPart=True, ReparaStd=Params['ReparaStd'], ModelSaveName=ModelLoadName)

        # Intermediate parameters 
        NFCs = SigRepModel.get_layer('FCs').output.shape[-1]


        # Setting Model Specifications and Sub-models
        if Params['LossType'] =='TCLosses':
            EncModel, FeatExtModel, FeatGenModel, ReconModel = ModelParts
        elif Params['LossType'] =='FACLosses':
            EncModel, FeatExtModel, FeatGenModel, ReconModel, FacDiscModel = ModelParts

        ## The generation model for evaluation
        RecOut = ReconModel(FeatGenModel.output)
        GenModel = Model(FeatGenModel.input, RecOut)

        ## The sampling model for evaluation
        Zs_Out = SigRepModel.get_layer('Zs').output
        SampModel = Model(EncModel.input, Zs_Out)



        #### -----------------------------------------------------  Conducting Evalution -----------------------------------------------------------------    
        # Instantiation 
        Eval = Evaluator(MinFreq = Params['MinFreq'], MaxFreq = Params['MaxFreq'], SimSize = Params['SimSize'], NMiniBat = Params['NMiniBat'], 
               NGen = Params['NGen'], ReparaStdZj = Params['ReparaStdZj'], NSelZ = Params['NSelZ'], SampBatchSize = Params['SampBatchSize'], 
               GenBatchSize = Params['GenBatchSize'], GPU = Params['GPU'])


        # The main task: Calculating and tracking the Conditional Mutual Information metric 
        ## FC_ArangeInp: A 2D matrix (NGen, NFCs) containing FCs values that the user creates and inputs directly.
        FC_ArangeInp = np.tile(np.linspace(Params['MinFreqR'], Params['MaxFreqR'], Params['NGen'])[:, None], (1, NFCs))

        ## SampZType: Z~ N(Zμ|y, σ) (SampZType = 'ModelRptA' or 'ModelRptB') vs. Z ~ N(0, ReparaStdZj) (SampZType = 'Gauss' or 'GaussRptA')
        Eval.Eval_ZFC(AnalData[:], SampModel, GenModel, FC_ArangeInp, FcLimit=Params['FcLimit'],  
                      WindowSize=Params['WindowSize'],  Continue=False, SampZType=Params['SampZType'])


        # Selecting post Samp_Zj for generating plausible signals
        PostSamp_Zj, NestedZFix = Eval.SelPostSamp_Zj( Params['MetricCut'], SavePath=SampZjSavePath)


        # Evaluating KLD (P || K)
        Eval.KLD_TrueGen(SecDataType ='FCR', RepeatSize = 1, PlotDist=False) 

        # Saving the instance's objects to a file
        SerializeObjects(Eval, Params['Common_Info']+Params['SpecZFC_Info'], ObjSavePath)


