import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from transformers.models.bert.modeling_tf_bert import TFBertEncoder,TFBertPooler,shape_list,get_initializer,shape_list
from transformers.models.bert import BertConfig

layers = tf.keras.layers

class Transformer_Blocks(tf.keras.Model):
    def __init__(self,use_decoder=False,layer_nums=2,heads=4,hidden_dim=256,head_hidden=512,activation='gelu',training=True,n_steps=5,**kwargs):
        super().__init__(**kwargs)
        self.config = BertConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=layer_nums,
            num_attention_heads=heads,
            intermediate_size=head_hidden,
            hidden_act=activation,
            use_cache=False,
            use_decoder=use_decoder
        )
        self.encoder =TFBertEncoder(config=self.config)    
        self.pooler = TFBertPooler(config=self.config)
        self.n_steps = n_steps

        self.input_embeddings = layers.Dense(hidden_dim,name='input_embed',kernel_initializer=get_initializer(self.config.initializer_range))
        self.pos_embeddings = self.add_weight(name='position_embed',shape=[n_steps,hidden_dim],initializer=get_initializer(self.config.initializer_range))
        self.input_dropout = layers.Dropout(self.config.hidden_dropout_prob)
        self.input_norm = layers.LayerNormalization()
    
    def input_embedds(self,state,training=True):
        input_shape = shape_list(state)[:-1]
        pos_idx = tf.expand_dims(tf.range(0,self.n_steps),axis=0)
        pos_embed = tf.gather(self.pos_embeddings,pos_idx)
        position_embeds = tf.tile(input=pos_embed, multiples=(input_shape[0], 1, 1))
        input_embed = self.input_embeddings(state)

        output = input_embed + position_embeds

        output = self.input_norm(output)
        output = self.input_dropout(output,training=training)

        return output

    
    def call(self,state,training=True):

        input_shape = shape_list(state)
        hidden_output = self.input_embedds(state)

        attn_mask = tf.ones(shape=input_shape[:-1])
        attn_mask = tf.reshape(attn_mask,[input_shape[0],1,1,input_shape[1]])
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            hidden_states=hidden_output,
            attention_mask=attn_mask,
            head_mask=head_mask,
            training=training,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=False
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(hidden_states=sequence_output)
        return (
                sequence_output,
                pooled_output,
            ) + encoder_outputs[1:]


class GaussianActor(tf.keras.Model):
    LOG_STD_CAP_MAX = 2  # np.e**2 = 7.389
    LOG_STD_CAP_MIN = -20  # np.e**-10 = 4.540e-05
    EPS = 1e-6

    def __init__(self, state_shape, action_dim, max_action, units=[256]*1,
                 hidden_activation="relu", state_independent_std=False,
                 squash=False, name='gaussian_policy',state_input=False,residual=False,lstm=False,trans=False):
        super().__init__(name=name)

        self._state_independent_std = state_independent_std
        self.lstm = lstm
        self._squash = squash
        self.residual = residual
        self.trans = trans

        # self.base_layers = []
        # for unit in units:
        #     self.base_layers.append(layers.Dense(unit, activation=hidden_activation))
        if not trans:
            if not state_input:
                self.conv_layers = [layers.Conv2D(16, 3, strides=3, activation='relu'), layers.Conv2D(64, 3, strides=2, activation='relu'), 
                                    layers.Conv2D(128, 3, strides=2, activation='relu'), layers.Conv2D(256, 3, strides=2, activation='relu'), 
                                    layers.GlobalAveragePooling2D()]
            else:
                self.conv_layers = []
                for unit in units:
                    self.conv_layers.append(layers.Dense(unit, activation=hidden_activation))
            if self.residual:
                self.norm_layers = [layers.LayerNormalization()]* len(self.conv_layers)
        
        

        self.connect_layers = [layers.Dense(128, activation='relu'), layers.Dense(32, activation='relu')]
        if self.lstm:
            self.lstm_layers = [layers.LSTM(256,return_sequences=True),layers.GlobalAveragePooling1D()]
            self.base_layers = self.lstm_layers + self.conv_layers +self.connect_layers
        elif self.trans:
            self.base_layers = self.connect_layers
            self.trans_layer = Transformer_Blocks()
        else:
            self.base_layers = self.conv_layers + self.connect_layers

        self.out_mean = layers.Dense(action_dim, name="L_mean")
        if self._state_independent_std:
            self.out_logstd = tf.Variable(
                initial_value=-0.5 * np.ones(action_dim, dtype=np.float32),
                dtype=tf.float32, name="L_logstd")
        else:
            self.out_logstd = layers.Dense(action_dim, name="L_logstd")

        self._max_action = max_action

        dummy_state = tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        self(dummy_state)

    def _compute_dist(self, states,test=False):
        """

        Args:
            states: np.ndarray or tf.Tensor
                Inputs to neural network.

        Returns:
            tfp.distributions.MultivariateNormalDiag
                Multivariate normal distribution object whose mean and
                standard deviation is output of a neural network
        """
        features = states
        
        if self.residual:
            for i, cur_layer in enumerate(self.base_layers):
                if i<len(self.conv_layers):
                    if i==0:
                        features = self.norm_layers[i](cur_layer(features))
                    else:
                        features = features + self.norm_layers[i](cur_layer(features))
                else:
                    features = cur_layer(features)
        else:
            if self.trans:
                # print(features.get_shape().as_list())
                pooled = self.trans_layer(features,training=1-test)
                features = pooled[1]
            for cur_layer in self.base_layers:
                features = cur_layer(features)

        mean = self.out_mean(features)

        if self._state_independent_std:
            log_std = tf.tile(
                input=tf.expand_dims(self.out_logstd, axis=0),
                multiples=[mean.shape[0], 1])
        else:
            log_std = self.out_logstd(features)
            log_std = tf.clip_by_value(log_std, self.LOG_STD_CAP_MIN, self.LOG_STD_CAP_MAX)

        return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.exp(log_std))

    def call(self, states, test=False):
        """
        Compute actions and log probabilities of the selected action
        """
        dist = self._compute_dist(states,test=test)
        if test:
            raw_actions = dist.mean()
        else:
            raw_actions = dist.sample()
        log_pis = dist.log_prob(raw_actions)
        entropy = dist.entropy()

        if self._squash:
            actions = tf.tanh(raw_actions)
            diff = tf.reduce_sum(tf.math.log(1 - actions ** 2 + self.EPS), axis=1)
            log_pis -= diff
        else:
            actions = raw_actions

        actions = actions * self._max_action
        return actions, log_pis,entropy

    def compute_log_probs(self, states, actions):
        raw_actions = actions / self._max_action
        dist = self._compute_dist(states)
        logp_pis = dist.log_prob(raw_actions)
        return logp_pis

    def compute_entropy(self, states):
        dist = self._compute_dist(states)
        return dist.entropy()
