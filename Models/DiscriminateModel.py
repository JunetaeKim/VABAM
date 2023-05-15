import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Dense,  Bidirectional 
from tensorflow.keras import Model



def DiscriminateModel (SigDim):
    InpL = Input(shape=(SigDim,), name='Inp_Enc')
    InpFrame = tf.signal.frame(InpL, 100, 100)
    DiscriLayer = Bidirectional(GRU(30, return_sequences=True))(InpFrame)
    DiscriLayer = Bidirectional(GRU(20, return_sequences=True))(DiscriLayer)
    DiscriLayer = Bidirectional(GRU(10, return_sequences=False))(DiscriLayer)
    DiscriLayer = Dense(20, activation='relu')(DiscriLayer)
    DiscriLayer = Dense(10, activation='relu')(DiscriLayer)
    DiscriLayer = Dense(5, activation='relu')(DiscriLayer)
    DiscriOut = Dense(1, activation='sigmoid')(DiscriLayer)
    
    Discriminator = Model(InpL, DiscriOut, name='Discriminator')
    Discriminator.compile(loss=tf.losses.binary_crossentropy, optimizer='adam')
    
    return Discriminator