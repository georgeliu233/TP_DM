
from transformers.models.bert.modeling_tf_bert import TFBertEncoder,TFBertPooler,shape_list,get_initializer,shape_list
from transformers.models.bert import BertConfig

import tensorflow as tf
import tf_geometric as tfg
import numpy as np

from tensorflow.keras.layers import Dense,Conv2D,LSTM,GlobalAveragePooling1D,RepeatVector,Dropout,LayerNormalization

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
    def __init__(self,lstm_layers=[256]*2,predicted_steps=5,predicted_dim=2):
        self.lstm_layers = [RepeatVector(predicted_steps)]
        for l in lstm_layers:
            self.lstm_layers.append(LSTM(l,return_sequences=True))
        self.lstm_layers.append(LSTM(predicted_dim,return_sequences=True))
    
    def call(self,hiddens):
        for layer in self.lstm_layers:
            hiddens = layer(hiddens)
        return hiddens
        
        
class Transformer_Blocks(tf.keras.Model):
    def __init__(self,use_decoder=False,layer_nums=2,heads=6,hidden_dim=256,head_hidden=768,activation='gelu',training=True,n_steps=5):
        self.config = BertConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=layer_nums,
            num_attention_heads=heads,
            intermediate_size=head_hidden,
            hidden_act=activation,
            use_cache=False,
            use_decoder=use_decoder
        )
        self.encoder =TFBertEncoder(self.config)    
        self.pooler = TFBertPooler(self.config)
        self.n_steps = n_steps

        self.input_embeddings = Dense(hidden_dim,name='input_embed',kernel_initializer=get_initializer(self.config.initializer_range))
        self.pos_embeddings = self.add_weight(name='position_embed',shape=[n_steps,hidden_dim],initializer=get_initializer(self.config.initializer_range))
        self.input_dropout = Dropout(self.config.hidden_dropout_prob)
        self.input_norm = LayerNormalization()
    
    def input_embedds(self,state,training=True):
        input_shape = shape_list(state)[:-1]
        pos_idx = tf.expand_dims(tf.range(0,self.n_steps),axis=0)
        pos_embed = tf.gather(self.pos_embeddings,pos_idx)
        position_embeds = tf.tile(input=position_embeds, multiples=(input_shape[0], 1, 1))
        input_embed = self.input_embeddings(state)

        output = input_embed + pos_embed

        output = self.input_norm(output)
        output = self.input_dropout(output,training=training)

        return output

    
    def call(self,state,training=True):

        input_shape = shape_list(state)
        hidden_output = self.input_embedds(state)

        attn_mask = tf.ones(dims=input_shape[:-1])
        attn_mask = tf.reshape(attn_mask,[input_shape[0],1,1,input_shape[1]])
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            hidden_states=hidden_output,
            attention_mask=attn_mask,
            head_mask=head_mask,
            training=training,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=False,
            training=training
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(hidden_states=sequence_output)
        return (
                sequence_output,
                pooled_output,
            ) + encoder_outputs[1:]









        
    
    
        

