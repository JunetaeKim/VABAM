import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Dense,  Bidirectional 
from tensorflow.keras import Model



def EvalDiscriminator (SigDim, SlidingSize = 50):
    InpL = Input(shape=(SigDim,), name='Inp_Enc')
    InpFrame = tf.signal.frame(InpL, SlidingSize, SlidingSize)
    DiscriLayer = Bidirectional(GRU(30, return_sequences=True))(InpFrame)
    DiscriLayer = Bidirectional(GRU(20, return_sequences=True))(DiscriLayer)
    DiscriLayer = Bidirectional(GRU(10, return_sequences=False))(DiscriLayer)
    DiscriLayer = Dense(20, activation='relu')(DiscriLayer)
    DiscriLayer = Dense(10, activation='relu')(DiscriLayer)
    DiscriLayer = Dense(5, activation='relu')(DiscriLayer)
    DiscriOut = Dense(1, activation='sigmoid')(DiscriLayer)
    
    EvalDiscModel = Model(InpL, DiscriOut, name='EvalDiscModel')
    EvalDiscModel.compile(loss=tf.losses.binary_crossentropy, optimizer='adam')
    
    return EvalDiscModel


def FacDiscriminator (LatDim, HDim):

    InpL = Input(shape=(LatDim,))
    DiscLayer = Dense(HDim, activation='relu')(InpL)
    DiscLayer = Dense(HDim, activation='relu')(DiscLayer)
    DiscLayer = Dense(HDim, activation='relu')(DiscLayer)
    DiscLayer = Dense(HDim, activation='relu')(DiscLayer)
    DiscLayer = Dense(HDim, activation='relu')(DiscLayer)
    DiscOut = Dense(2, activation='linear')(DiscLayer)
    FacDiscModel = Model(InpL, DiscOut, name='FacDiscModel')
    
    return  FacDiscModel