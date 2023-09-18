import sys
# setting path
sys.path.append('../')

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import copy
from argparse import ArgumentParser

from Benchmarks.Models.BenchmarkCaller import *
from Utilities.EvaluationMain import *
from Utilities.Utilities import ReadYaml, SerializeObjects, DeserializeObjects, LoadModelConfigs, LoadParams

# python .\BatchBMEvaluation.py --Config EvalConfigART --GPUID 0 
    

if __name__ == "__main__":

    
    # Create the parser
    parser = ArgumentParser()
    
    # Add Experiment-related parameters
    parser.add_argument('--Config', type=str, required=True, help='Set the name of the configuration to load (the name of the YAML file).')
    parser.add_argument('--ConfigSpec', nargs='+', type=str, required=False, 
                        default=None, help='Set the name of the specific configuration to load (the name of the model config in the YAML file).')
    parser.add_argument('--GPUID', type=int, required=False, default=1)
    
    args = parser.parse_args() # Parse the arguments
    ConfigName = args.Config
    ConfigSpecName = args.ConfigSpec
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
        ModelConfigSet, ModelLoadPath = LoadModelConfigs(ConfigName, Comp=False)

        # Loading parameters for the evaluation
        Params = LoadParams(ModelConfigSet, EvalConfigs[ConfigName])
        Params['Common_Info'] = EvalConfigs['Common_Info']

        # Object save path
        ObjSavePath = './EvalResults/Instances/Obj_'+ConfigName+'.pkl'
        SampZjSavePath = './Data/IntermediateData/'+ConfigName+'_SampZj_'+str(Params['NSelZ'])+'.npy'


        #### -----------------------------------------------------   Loading data -------------------------------------------------------------------------   
        if SigTypePrev != Params['SigType']:
            SigTypePrev = Params['SigType'] # To change data type: ART, II, PLETH

            # Loading data
            TrData = np.load('../Data/ProcessedData/Tr'+str(Params['SigType'])+'.npy').astype('float32')
            VallData = np.load('../Data/ProcessedData/Val'+str(Params['SigType'])+'.npy').astype('float32')

        # Intermediate parameters 
        SigDim = VallData.shape[1]
        DataSize = VallData.shape[0]


        
        #### -----------------------------------------------------   Defining model structure -----------------------------------------------------------------    
        # Calling Modesl
        BenchModel, _, AnalData = ModelCall (ModelConfigSet, ConfigName, TrData, VallData, LoadWeight=True,  
                                                 Reparam=True, ReparaStd=Params['ReparaStd'], ModelSaveName=ModelLoadPath) 
        
        
        ## The generation model for evaluation
        GenModel = BenchModel.get_layer('ReconModel')
        
        ## The sampling model for evaluation
        Inp_Enc = BenchModel.get_layer('Inp_Enc')
        Zs = BenchModel.get_layer('Zs').output
        
        if Params['SecDataType'] == 'CONA' or Params['SecDataType'] == 'CONR':
            Inp_Cond = BenchModel.get_layer('Inp_Cond')
            SampModel = Model([Inp_Enc.input, Inp_Cond.input], Zs)
        else:
            SampModel = Model(Inp_Enc.input, Zs)


        #### -----------------------------------------------------  Conducting Evalution -----------------------------------------------------------------    
        # Instantiation 
        Eval = Evaluator(MinFreq = Params['MinFreq'], MaxFreq = Params['MaxFreq'], SimSize = Params['SimSize'], NMiniBat = Params['NMiniBat'], 
               NGen = Params['NGen'], ReparaStdZj = Params['ReparaStdZj'], NSelZ = Params['NSelZ'], SampBatchSize = Params['SampBatchSize'], 
               GenBatchSize = Params['GenBatchSize'], GPU = Params['GPU'])
        
        if Params['SecDataType'] == 'CONA' or Params['SecDataType'] == 'CONR':
            ## SampZType: Z~ N(Zμ|y, σ) (SampZType = 'ModelRptA' or 'ModelRptB') vs. Z ~ N(0, ReparaStdZj) (SampZType = 'Gauss' or 'GaussRptA')
            Eval.Eval_ZCON(AnalData,  SampModel, GenModel, Continue=False, WindowSize=Params['WindowSize'],
                           SampZType=Params['SampZType'], SecDataType=Params['SecDataType'])
        else:
            Eval.Eval_Z(AnalData, SampModel, GenModel,  Continue=False, SampZType=Params['SampZType'])


        # Selecting post Samp_Zj for generating plausible signals
        SelPostSamp = Eval.SelPostSamp( Params['MetricCut'], SavePath=SampZjSavePath)


        # Evaluating KLD (P || K)
        Eval.KLD_TrueGen(SecDataType = Params['SecDataType'], RepeatSize = 1, PlotDist=False) 

        # Saving the instance's objects to a file
        SerializeObjects(Eval, Params['Common_Info']+Params['Spec_Info'], ObjSavePath)


