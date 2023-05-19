import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K



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
    def __init__(self, BetaList, LossScaling, MinLimit , MaxLimit , verbose=1, ToSaveLoss=None, SaveWay=None, SaveLogOnly=True, SavePath=None, CheckPoint=False):
            
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
        self.Logs = []
        self.SaveLogOnly = SaveLogOnly
        self.LogsPath = "./Logs_" + SavePath.split('/')[-1].split('.')[0] + '.txt'
        self.CheckPoint = CheckPoint
        
        
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
            
         
        
class RandFCs(tf.keras.layers.Layer):
    
    def __init__(self, ):
        super(RandFCs, self).__init__(name='FCs')
        pass
    
    def build(self, input_shape):
        self.GenVec = tf.Variable(tf.random.uniform(shape=(1,6)), trainable=False)
    
    def call(self, input):
        return tf.tile(self.GenVec , (tf.shape(input)[0],1))