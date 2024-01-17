import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
import yaml
import pickle
import re


def ReadYaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    

    
def ReName (layer, name):
    return Lambda(lambda x: x, name=name)(layer)



def CompResource (PredModel, Data, BatchSize=1, GPU=True):  # GPU vs CPU

    if GPU==False:
        with tf.device('/CPU:0'):
            PredVal = PredModel.predict(Data, batch_size=BatchSize, verbose=1)
    else:
        PredVal = PredModel.predict(Data, batch_size=BatchSize, verbose=1)

    return PredVal



def LoadModelConfigs(ConfigName, Training=True, Comp=True, RootDirYaml=None, RootDirRes=None):
    
    # Whether the model performs signal compression (i.e., the main model) or not.
    CompSize = re.findall(r'\d+', ConfigName)[-1] if Comp else ''

    # Set default values for RootDirYaml and RootDirRes if they are None
    if RootDirYaml is None:
        RootDirYaml = './Config/'
    if RootDirRes is None:
        RootDirRes = './Results/'
            
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
        assert False, "Please verify if the data type is properly included in the name of the configuration. The configuration name should be structured as 'Config' + 'data type' + CompSize(optional) , such as ConfigART800."

    YamlPath = RootDirYaml + LoadConfig+'.yml'
    ConfigSet = ReadYaml(YamlPath) # The YAML file
        
    ### Model path
    ModelName = ConfigName +'.hdf5'
    ModelPath = RootDirRes + SubPath + ModelName
   
    ### Checking whether the model path exists or not.
    if not os.path.exists(RootDirRes + SubPath) and Training == True:
        os.makedirs(RootDirRes + SubPath)
    
    return ConfigSet[ConfigName], ModelPath



# Serialize the objects
def SerializeObjects(Instance, SaveResList, Filename):
    """
    Serialize the objects of an Evaluator instance to a file.

    Parameters:
    - Instance: The instance of the Evaluator class.
    - Filename: The name of the file where objects will be saved.
    """

    DataToSave = {Name: getattr(Instance, Name) for Name in SaveResList}

    with open(Filename, "wb") as F:
        pickle.dump(DataToSave, F)

        
        
# Deserialize the objects
def DeserializeObjects(Instance, Filename):
    """
    Deserialize objects from a file and set them to the attributes of the provided Evaluator instance.

    Parameters:
    - Instance: The instance of the Evaluator class.
    - Filename: The name of the file from where objects will be loaded.
    """
    with open(Filename, "rb") as F:
        LoadedData = pickle.load(F)

    for Name, Value in LoadedData.items():
        setattr(Instance, Name, Value)
        

        
def LoadParams (ModelConfigSet, EvalConfigSet): # Experiment setting

    Params = {}
    
    ### Model-related parameters
    Params['SigType']  = ModelConfigSet['SigType']            # Types of signals to train on.: ART, PLETH, II. 
    Params['LatDim']  = ModelConfigSet['LatDim']              # The dimensionality of the latent variable z.
    Params['ReparaStd'] = EvalConfigSet['ReparaStd']          # The standard deviation value for Gaussian noise generation used in the reparametrization trick.
    Params['ReparaStdZj'] = EvalConfigSet['ReparaStdZj']      # The standard deviation when sampling Zj (Samp_ZjRPT ~ N(0, ReparaStdZj)).

    
    ### Evaluation-related parameters
    Params['MaxFreq'] = EvalConfigSet['MaxFreq']              # The maximum frequency value within the analysis range (default = 51).
    Params['MinFreq'] = EvalConfigSet['MinFreq']              # The minimum frequency value within the analysis range (default = 1).
    Params['NMiniBat'] = EvalConfigSet['NMiniBat']            # The size of the mini-batch, splitting the task into N pieces of size NMiniBat.
    Params['SimSize'] = EvalConfigSet['SimSize']              # The number of generations (i.e., samplings) within the mini-batch.
    Params['NGen'] = EvalConfigSet['NGen']                    # The number of generations (i.e., samplings) within the mini-batch.
    Params['NParts'] = EvalConfigSet['NParts']                    # The number of partitions (i.e., samplings) in generations within a sample.
    Params['NSelZ'] = EvalConfigSet['NSelZ']                  # The size of js to be selected at the same time (default: 1).
    Params['SelMetricType'] = EvalConfigSet['SelMetricType']  # The type of metric used for selecting Zs and ancillary data. 
    Params['SelMetricCut'] = EvalConfigSet['SelMetricCut']    # The threshold value for selecting Zs whose Entropy or KLD of PSD is less than the MetricCut.
    Params['SecDataType'] = EvalConfigSet['SecDataType']      # The secondary data type
        
    ### Functional parameters
    Params['SampBatchSize'] = EvalConfigSet['SampBatchSize']  # The batch size during prediction of the sampling model.
    Params['GenBatchSize'] = EvalConfigSet['GenBatchSize']    # The batch size during prediction of the generation model.
    Params['GPU'] = EvalConfigSet['GPU']                      # GPU vs CPU during model predictions (i.e., for SampModel and GenModel).
    Params['Spec_Info'] = EvalConfigSet['Spec_Info']          # The list of specific objects subject to class serialization.
     
    
    ### Model-specific parameters
    if 'WindowSize' in EvalConfigSet:
        Params['WindowSize'] = EvalConfigSet['WindowSize']    # The window size when calculating permutation entropy (default: 3)
    if 'FcLimit' in EvalConfigSet:
        Params['FcLimit'] = EvalConfigSet['FcLimit']          # The threshold value of the max of the FC value input into the generation model.
        
        
    ### Loss-specific parameters (only for main models)
    if 'LossType' in ModelConfigSet:
        Params['LossType'] = ModelConfigSet['LossType']
    if 'SpecLosses' in ModelConfigSet:
        Params['SpecLosses'] = ModelConfigSet['SpecLosses']

    
    return Params        
        
        

        
class Lossweight(tf.keras.layers.Layer):
    
    def __init__(self, InitVal = 0., name='Lossweight'):
        super(Lossweight, self).__init__(name=name)
        self.InitVal = InitVal
    
    def build(self, input_shape):
        self.GenVec = tf.Variable(self.InitVal, trainable=False)
    
    def call(self, input):

        return self.GenVec
    
    def get_config(self):
        config = super(Lossweight, self).get_config()
        config.update({ 'InitVal': self.InitVal })
        return config
    
    


class Anneal(tf.keras.callbacks.Callback):
    def __init__(self, TargetLossName, Threshold,  BetaName, MaxBeta=0.1, MinBeta=1e-5, AnnealEpoch=100, UnderLimit=0., verbose=1):
        
        '''
        if type(TargetLossName) != list:
            TargetLossName = [TargetLossName]
        '''
        #KLD_Beta_Z = Anneal(TargetLossName=['val_FeatRecLoss', 'val_RecLoss'], Threshold=0.001, BetaName='Beta_Z',  MaxBeta=0.1 , MinBeta=0.1, AnnealEpoch=1, UnderLimit=1e-7, verbose=2)
        #KLD_Beta_Fc = Anneal(TargetLossName=['val_FeatRecLoss', 'val_RecLoss'], Threshold=0.001, BetaName='Beta_Fc',  MaxBeta=0.005 , MinBeta=0.005, AnnealEpoch=1, UnderLimit=1e-7, verbose=1)

        self.TargetLossName = TargetLossName
        self.Threshold = Threshold
        self.BetaName = BetaName
        self.AnnealIdx = 0
        self.verbose = verbose 
        self.Beta =  np.concatenate([np.array([UnderLimit]), np.linspace(start=MinBeta, stop=MaxBeta, num=AnnealEpoch+1 )])

    def on_epoch_end(self, epoch, logs={}):
        
        #TargetLoss = max([logs[i] for i in self.TargetLossName.keys()]) 
        TargetLoss = max([logs[LossName] / np.maximum(1e-7,self.model.get_layer(BetaName).variables[0].numpy())  for LossName, BetaName in self.TargetLossName.items()]) 
        
        
        if TargetLoss > self.Threshold:
            
            self.AnnealIdx -= 1
            self.AnnealIdx = np.maximum(self.AnnealIdx, 0)
            K.set_value(self.model.get_layer(self.BetaName).variables[0], self.Beta[self.AnnealIdx])
        else: 
            self.AnnealIdx += 1
            self.AnnealIdx = np.minimum(self.AnnealIdx, len(self.Beta)-1)

            K.set_value(self.model.get_layer(self.BetaName).variables[0], self.Beta[self.AnnealIdx])
        
        if self.verbose==1:
            print(self.BetaName+' : ' ,self.model.get_layer(self.BetaName).variables[0].numpy())
        elif self.verbose==2:
            print('TargetLoss : ', TargetLoss)
            print(self.BetaName+' : ' ,self.model.get_layer(self.BetaName).variables[0].numpy())
            
          

        
class RelLossWeight(tf.keras.callbacks.Callback):
    def __init__(self, BetaList, LossScaling, MinLimit , MaxLimit , SavePath, verbose=1, ToSaveLoss=None, SaveWay=None, SaveLogOnly=True,  CheckPoint=False, Resume=False):
            
        if type(ToSaveLoss) != list and ToSaveLoss is not None:
            ToSaveLoss = [ToSaveLoss]
            
        self.BetaList = BetaList
        self.LossScaling = LossScaling
        self.MinLimit = MinLimit
        self.MaxLimit = MaxLimit
        self.verbose = verbose
        self.ToSaveLoss = ToSaveLoss
        self.CheckLoss = np.inf
        self.SaveWay = SaveWay
        self.SavePath = SavePath
        self.Resume = Resume
        PathInfo = SavePath.split('/')
        self.ModelName = PathInfo[-1].split('.')[0]
        self.SaveLogOnly = SaveLogOnly
        self.LogsPath = './Logs/'+PathInfo[2]+'/Logs_'+self.ModelName+ '.txt'
        self.CheckPoint = CheckPoint

        if Resume == True:
            with open(self.LogsPath, "r") as file:
                self.Logs = file.read().split('\n')
                self.StartEpoch = int(self.Logs[-1].split(' ')[0]) + 1
        else:
            self.Logs = []
            self.StartEpoch = 0
        
        if not os.path.exists('./Logs/'+PathInfo[2]):
            os.makedirs('./Logs/'+PathInfo[2])
        
        
    def on_epoch_end(self, epoch, logs={}):
        
        Losses = {key:logs[key]/np.maximum(1e-7,self.model.get_layer(beta).variables[0].numpy()) for key, beta in self.BetaList.items()}
        rounded_losses = {key: round(value, 5) for key, value in Losses.items()}

        if self.SaveLogOnly and self.ToSaveLoss == None:
            self.Logs.append(str(epoch+self.StartEpoch)+' '+ str(rounded_losses))
            
            with open(self.LogsPath, "w") as file:
                file.write('\n'.join(self.Logs))
        
        if self.CheckPoint:
            if (epoch+self.StartEpoch) % self.CheckPoint==0:
                CharIDX = self.SavePath.rfind('/')
                Path = self.SavePath[:CharIDX+1] +'Epoch' + str(epoch+self.StartEpoch)+'_' + self.SavePath[CharIDX+1:]
                print(Path)
                self.model.save(Path)
        
        
        if self.ToSaveLoss is not None:
            
            SubLosses = []
            for LossName in self.ToSaveLoss:
                beta = self.BetaList[LossName]
                SubLosses.append(logs[LossName]/np.maximum(1e-7,self.model.get_layer(beta).variables[0].numpy()))
            
            if self.SaveWay == 'min':
                CurrentLoss = np.min(SubLosses)
            elif self.SaveWay == 'mean':
                CurrentLoss = np.mean(SubLosses)
            else:
                CurrentLoss = np.max(SubLosses)
            
            if CurrentLoss <= self.CheckLoss and (epoch+self.StartEpoch) > 0:
                self.model.save(self.SavePath)
                print()
                print('The model has been saved since loss decreased from '+ str(self.CheckLoss)+ ' to ' + str(CurrentLoss))
                print()
                
                #self.Logs.append(str(epoch+self.StartEpoch)+': The model has been saved since loss decreased from '+ str(self.CheckLoss)+ ' to ' + str(CurrentLoss))
                self.Logs.append(str(epoch+self.StartEpoch)+' saved '+ str(rounded_losses))
                
                with open(self.LogsPath, "w") as file:
                    file.write('\n'.join(self.Logs))
                
                self.CheckLoss = CurrentLoss
                
            elif (epoch+self.StartEpoch) > 0:
                print()
                print('The model has not been saved since the loss did not decrease.')
                print()
                
                #self.Logs.append(str(epoch+self.StartEpoch)+': The model has not been saved since the loss did not decrease from '+ str(CurrentLoss)+ ' to ' + str(self.CheckLoss))
                self.Logs.append(str(epoch+self.StartEpoch)+' not saved '+ str(rounded_losses))
                
                with open(self.LogsPath, "w") as file:
                    file.write('\n'.join(self.Logs))
                
                
                
        WeigLosses = np.maximum(1e-7, np.array(list(Losses.values())))
        #print(WeigLosses)
        RelWeights = WeigLosses / np.min(WeigLosses)
        #print(RelWeights)
        RelWeights = {loss:RelWeights[num] * self.LossScaling[loss] for num, loss in enumerate (self.BetaList.keys())}
        #print(RelWeights)

        for loss, beta in self.BetaList.items():
            
            Value = np.clip(RelWeights[loss] , self.MinLimit[beta], self.MaxLimit[beta])
            K.set_value(self.model.get_layer(beta).variables[0], Value)

        if self.verbose==1:

            print('----------------' +self.ModelName+ '--------------------')
            print('Losses')
            for key, value in Losses.items():
                print("%s: %.7f" % (key, value))
            
            print('------------------------------------')
            print('RelWeights')
            for key, value in RelWeights.items():
                print("%s: %.7f" % (key, value))

            print('------------------------------------')
            print('Beta')
            for key, beta in self.BetaList.items():
                print(beta, ': ', self.model.get_layer(beta).variables[0].numpy())
            print('------------------------------------')
            print()
            
  
    
