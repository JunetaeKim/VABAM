import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
import yaml



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
    def __init__(self, BetaList, LossScaling, MinLimit , MaxLimit , SavePath, verbose=1, ToSaveLoss=None, SaveWay=None, SaveLogOnly=True,  CheckPoint=False):
            
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
        PathInfo = SavePath.split('/')
        self.SaveLogOnly = SaveLogOnly
        self.LogsPath = './Logs/'+PathInfo[2]+'/Logs_'+PathInfo[-1].split('.')[0]+ '.txt'
        self.CheckPoint = CheckPoint
        self.Logs = []
        
        if not os.path.exists('./Logs/'+PathInfo[2]):
            os.mkdir('./Logs/'+PathInfo[2])
        
        
    def on_epoch_end(self, epoch, logs={}):
        
        Losses = {key:logs[key]/np.maximum(1e-7,self.model.get_layer(beta).variables[0].numpy()) for key, beta in self.BetaList.items()}
        rounded_losses = {key: round(value, 5) for key, value in Losses.items()}

        if self.SaveLogOnly and self.ToSaveLoss == None:
            self.Logs.append(str(epoch)+' '+ str(rounded_losses))
            
            with open(self.LogsPath, "w") as file:
                file.write('\n'.join(self.Logs))
        
        if self.CheckPoint:
            if epoch % self.CheckPoint==0:
                CharIDX = self.SavePath.rfind('/')
                Path = self.SavePath[:CharIDX+1] +'Epoch' + str(epoch)+'_' + self.SavePath[CharIDX+1:]
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
            
            if CurrentLoss <= self.CheckLoss and epoch > 0:
                self.model.save(self.SavePath)
                print()
                print('The model has been saved since loss decreased from '+ str(self.CheckLoss)+ ' to ' + str(CurrentLoss))
                print()
                
                #self.Logs.append(str(epoch)+': The model has been saved since loss decreased from '+ str(self.CheckLoss)+ ' to ' + str(CurrentLoss))
                self.Logs.append(str(epoch)+' saved '+ str(rounded_losses))
                
                with open(self.LogsPath, "w") as file:
                    file.write('\n'.join(self.Logs))
                
                self.CheckLoss = CurrentLoss
            elif epoch > 0:
                print()
                print('The model has not been saved since the loss did not decrease from '+ str(CurrentLoss)+ ' to ' + str(self.CheckLoss))
                print()
                
                #self.Logs.append(str(epoch)+': The model has not been saved since the loss did not decrease from '+ str(CurrentLoss)+ ' to ' + str(self.CheckLoss))
                self.Logs.append(str(epoch)+' not saved '+ str(rounded_losses))
                
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

            print('------------------------------------')
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
            
  
    
def log_importance_weight_matrix(batch_size, dataset_size):
    """
    Calculates a log importance weight matrix

    Parameters
    ----------
    batch_size: int
        number of training images in the batch

    dataset_size: int
        number of training images in the dataset
        
    Reference:
    https://github.com/JunetaeKim/disentangling-vae-torch/blob/master/disvae/utils/math.py#L54
    """
    
    
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)

    W = tf.fill([batch_size, batch_size], 1 / M)
    W = tf.reshape(W, [-1])
    W = tf.tensor_scatter_nd_update( W,  tf.range(0, tf.size(W), M + 1)[:, None],  tf.fill([tf.size(W) // (M + 1)], 1 / N))
    W = tf.tensor_scatter_nd_update( W, tf.range(1, tf.size(W), M + 1)[:, None], tf.fill([tf.size(W) // (M + 1) - 1 + 1], strat_weight))
    W = tf.tensor_scatter_nd_update( W,  tf.constant([[M * (M + 1)]]),   tf.constant([strat_weight]))
    W = tf.reshape(W, [batch_size, batch_size])

    return tf.math.log(W)