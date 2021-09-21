import tensorflow as tf
import tf_geometric as tfg
import numpy as np

from tensorflow.keras.layers import Dense,Conv2D,LSTM,GlobalAveragePooling1D

class LSTM_GAT_Encoder(tf.keras.Model):
    def __init__(self,gat_layers=[256,4]*2,lstm_layers=[256]*2,num_of_vehicles=1+3,undirect_edge=True):
        
        self.num_of_vehicles = num_of_vehicles
        self.undirect_edge = undirect_edge
        self.lstm_layers = []

        for l in lstm_layers:
            self.lstm_layers.append(LSTM(l,return_sequences=True))
        self.lstm_layers.append(GlobalAveragePooling1D())

        self.gat_layers = []

        #we build a star graph (directed?)
        for units,head in gat_layers:
            self.gat_layers.append(tfg.layers.GAT(units=units,num_heads=head,activation=tf.nn.gelu))
        
        u,v = list(np.arange(self.num_of_vehicles)),[0]*self.num_of_vehicles

        if undirect_edge:
            self.graph_index = np.array(u+v,v+u)
        else:
            self.graph_index = np.array(u,v)

        
    def call(self,states):
        """
        suppose the ego car in the middle
        we build a star graph (directed?)

        states:[batch,num_of_vehicles,timestep,input_dim]

        lstm_out:[batch,num_of_vehicles,lstm_hidden]

        gat_out:[batch,num_of_vehicles,gat_units]
        """
        for layer in self.lstm_layers:
            states = layer(states)
        
        for g_layer in self.gat_layers:
            states = g_layer([states,self.graph_index])
        

class CNN_Encoder(tf.keras.Model):
    def __init__(self,lstm_layers=[256]*2):

        self.conv_layers = [layers.Conv2D(16, 3, strides=3, activation='relu'), layers.Conv2D(64, 3, strides=2, activation='relu'), 
                                layers.Conv2D(128, 3, strides=2, activation='relu'), layers.Conv2D(256, 3, strides=2, activation='relu'), 
                                layers.GlobalAveragePooling2D()]
        self.lstm_layers = []
        for l in lstm_layers:
            self.lstm_layers.append(LSTM(l,return_sequences=True))
        self.lstm_layers.append(GlobalAveragePooling1D())
    
    def call(self,imgs):
        for layer in self.conv_layers:
            imgs = layer(imgs)
        
        for layer in self.lstm_layers:
            imgs = layer(imgs)
        
        return imgs
    
class Traj_decoder(tf.keras.Model):
        
        
    
    
        

