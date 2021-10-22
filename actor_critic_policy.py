import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
layers = tf.keras.layers

class Ego_Neighbours_Encoder(tf.keras.layers.Layer):
    def __init__(self, state_shape,name='ego_surr_encode',units=256,
        use_trans_encode=False,num_heads=6,drop_rate=0.1,neighbours=5,
        make_rotation=True,time_step=8):
        super().__init__(name=name)
        self.neighbours = neighbours
        self.make_rotation=make_rotation
        self.time_step=time_step
        self.lstm_layer = layers.GRU(units,return_sequences=True)
        self.use_trans = use_trans_encode
        self.ego_norm = layers.LayerNormalization()
        if use_trans_encode:
            self.rel_layer = layers.MultiHeadAttention(num_heads,units,dropout=drop_rate)
            self.FFN_layers = [
                layers.Dense(4*units,activation='gelu'),
                layers.Dense(units)
            ]
            self.dropout_layers = [
                layers.Dropout(drop_rate)
            ]*len(self.FFN_layers)

            self.norm_layers = [
                layers.LayerNormalization()
            ]*2
        else:
            self.encode_layer = layers.Dense(
                units,activation='gelu'
            )
    
    def wrap_to_pi(self,theta):
        pi = tf.constant(np.pi)
        return (theta+pi) % (2*pi) - pi

    def get_speed(self,inputs):
        #[batch,ego+neigh,time_steps]

        #zeros padding[batch,ego+neigh,1]
        pad = tf.expand_dims(tf.zeros_like(inputs[:,:,0]),2)
        speed = (inputs[:,:,1:] - inputs[:,:,:-1])/0.1
        return tf.concat([pad,speed], axis=-1)

    def _make_rotations(self,states,mask):
        """
        ref : lhc_mtp_loss.py 
        make clock-wise rotation after moving to the ego point
        input states :[batch,ego+5*neighbours,timesteps,hidden]

        """
        #get curr_frame location according to the mask
        ind = tf.reduce_sum(mask,axis=-1)

        #gather indices is [batch_no , 0(ego) , mask_ind]
        #return [batch,hidden(4)]
        curr_frames = tf.gather_nd(states, tf.transpose([
            tf.range(mask.get_shape()[0]), tf.zeros_like(ind) , ind
        ]))

        yaw = curr_frames[:,-1]
        cos_a = tf.reshape(tf.math.cos(yaw),[-1,1,1])
        sin_a = tf.reshape(tf.math.sin(yaw),[-1,1,1])

        #[batch,6,time_steps]
        x = states[:,:,:,0] - tf.reshape(curr_frames[:,0],[-1,1,1])
        y = states[:,:,:,1] - tf.reshape(curr_frames[:,1],[-1,1,1])
        dis = states[:,:,:,2] - tf.reshape(curr_frames[:,1],[-1,1,1])
        angle = self.wrap_to_pi(states[:,:,:,3]-tf.reshape(yaw,[-1,1,1]))

        new_x =tf.multiply(cos_a,x) - tf.multiply(sin_a,y)
        new_y =tf.multiply(sin_a,x) + tf.multiply(cos_a,y)
        vx = self.get_speed(new_x)
        vy = self.get_speed(new_y)

        rotated_state = [
            tf.expand_dims(new_x,3),
            tf.expand_dims(new_y,3),
            tf.expand_dims(vx,3),
            tf.expand_dims(vy,3),
            tf.expand_dims(dis,3),
            tf.expand_dims(angle,3)
        ]
        return tf.concat(rotated_state, axis=-1)

    def _split_ego(self,states,mask):

        states = tf.reshape(states, [-1,self.time_step,self.neighbours+1,4])
        states = tf.transpose(states,[0,2,1,3])

        if self.make_rotation:
            states = self._make_rotations(states,mask)
        else:
            x = states[:,:,:,0]
            y = states[:,:,:,1]
            vx = self.get_speed(x)
            vy = self.get_speed(y)
            dis = states[:,:,:,2]
            angle = self.wrap_to_pi(states[:,:,:,3])
            rotated_state = [
            tf.expand_dims(new_x,3),
            tf.expand_dims(new_y,3),
            tf.expand_dims(vx,3),
            tf.expand_dims(vy,3),
            tf.expand_dims(dis,3),
            tf.expand_dims(angle,3)
            ]
            states = tf.concat(rotated_state, axis=-1)

        e_states,n_states = states[:,0,:,:] , states[:,1:,:,:]
        
        return e_states , n_states

    def _lstm_with_mask(self,states,mask):

        full_states = self.lstm_layer(inputs=states)
        ind = tf.reduce_sum(mask,axis=-1)
        states = tf.gather_nd(full_states, tf.transpose([
            tf.range(mask.get_shape()[0]),ind
        ]))

        return states

    def call(self,states,mask,test=False):

        mask = tf.cast(mask,tf.int32)
        training= bool(1-test)
        
        ego_states , neighbor_states = self._split_ego(states,mask)
        actor_mask = tf.not_equal(tf.concat([tf.expand_dims(ego_states, 1), neighbor_states], axis=1), 0)[:, :, 0, 0]

        ego = self._lstm_with_mask(ego_states,mask)
        neighbors =[
            self._lstm_with_mask(neighbor_states[:,i,:,:],mask) for i in range(self.neighbours)
        ]

        actor = tf.concat([ego[:, tf.newaxis], tf.stack(neighbors, axis=1)], axis=1)

        if self.use_trans:
            value = self.rel_layer(tf.expand_dims(ego, axis=1), actor,attention_mask=actor_mask[:, tf.newaxis], training=training)
            value = tf.squeeze(value,axis=1)
            value = self.norm_layers[0](value)

            for i in range(len(self.FFN_layers)):
                value = self.FFN_layers[i](value)
                value = self.dropout_layers[i](value,training=training)
            
            value = self.norm_layers[-1](value)
        else:
            value = self.encode_layer(actor)
        
        
        
        feature = tf.concat([value,self.ego_norm(ego)],axis=-1)
        return feature
    
        

class GaussianActorCritic(tf.keras.Model):
    LOG_STD_CAP_MAX = 2  # np.e**2 = 7.389
    LOG_STD_CAP_MIN = -20  # np.e**-10 = 4.540e-05
    EPS = 1e-6

    def __init__(self, state_shape, action_dim, max_action, units=[256]*3,
                 hidden_activation="relu", state_independent_std=False,
                 squash=False, name='gaussian_policy',state_input=False,residual=False,lstm=False,trans=False,
                 cnn_lstm=False,bptt=False,ego_surr=False,use_trans=False,neighbours=5,time_step=8):
        super().__init__(name=name)

        self._state_independent_std = state_independent_std
        self.lstm = lstm
        self.cnn_lstm = cnn_lstm
        self.bptt=bptt
        self.ego_surr=ego_surr

        self._squash = squash
        self.residual = residual
        self.trans = trans

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
                #CNN
                self.encode_layers = [layers.Conv2D(16, 3, strides=3, activation='relu'), layers.Conv2D(64, 3, strides=2, activation='relu'), 
                                    layers.Conv2D(128, 3, strides=2, activation='relu'), layers.Conv2D(256, 3, strides=2, activation='relu'), 
                                    layers.GlobalAveragePooling2D()]
            elif lstm:
                self.lstm_layers = layers.GRU(256,return_sequences=True)
                print('lstm')
                self.encode_layers = [
                    layers.Dense(256)]
            elif cnn_lstm:
                self.cnn_layers = [layers.Conv2D(16, 3, strides=3, activation='relu'), layers.Conv2D(64, 3, strides=2, activation='relu'), 
                                    layers.Conv2D(128, 3, strides=2, activation='relu'), layers.Conv2D(256, 3, strides=2, activation='relu'), 
                                    layers.GlobalAveragePooling2D(),layers.LSTM(256,return_sequences=True)]
                self.encode_layers = [layers.GlobalAveragePooling1D(),
                                    layers.Dense(256,activation='relu')]   
                self.encode_layers = self.cnn_layers+self.encode_layers
            elif ego_surr:
                self.ego_layer = Ego_Neighbours_Encoder(state_shape,use_trans_encode=use_trans,
                neighbours=neighbours,time_step=time_step,num_heads=6)
            else:
                self.encode_layers = []
                for unit in units:
                    self.encode_layers.append(layers.Dense(unit, activation=hidden_activation))
        else:
            pass
        
        self.actor_layers = [layers.Dense(128, activation='relu'), 
                                layers.Dense(32, activation='relu')]
        self.critic_layers = [layers.Dense(128, activation='relu'), 
                                layers.Dense(32, activation='relu'),
                                layers.Dense(1)]

        self.out_mean = layers.Dense(action_dim, name="L_mean")
        if self._state_independent_std:
            self.out_logstd = tf.Variable(
                initial_value=-0.5 * np.ones(action_dim, dtype=np.float32),
                dtype=tf.float32, name="L_logstd")
        else:
            self.out_logstd = layers.Dense(action_dim, name="L_logstd")

        self._max_action = max_action
        print(state_shape)
        dummy_state = tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        mask = tf.ones([1,dummy_state.get_shape()[1]])
        print(mask.get_shape())
        self(dummy_state,mask)
        self.summary()
    
    def _compute_dist(self, features,test=False):
        """

        Args:
            states: np.ndarray or tf.Tensor
                Inputs to neural network.

        Returns:
            tfp.distributions.MultivariateNormalDiag
                Multivariate normal distribution object whose mean and
                standard deviation is output of a neural network
        """
        for layer in self.actor_layers:
            features = layer(features)

        mean = self.out_mean(features)

        if self._state_independent_std:
            log_std = tf.tile(
                input=tf.expand_dims(self.out_logstd, axis=0),
                multiples=[mean.shape[0], 1])
        else:
            log_std = self.out_logstd(features)
            log_std = tf.clip_by_value(log_std, self.LOG_STD_CAP_MIN, self.LOG_STD_CAP_MAX)

        return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.exp(log_std))

    def _state_encoding(self,states,mask, test=False):
        # print(states)
        # if self.cnn_lstm:
        #     for layer in self.cnn_layers:
        #         states = layer(states)
        if self.ego_surr:
            states = self.ego_layer(states,mask,test)
        else:
            if self.lstm:
                # print(mask.get_shape())
                # print(states.get_shape())
                mask = tf.cast(mask,tf.int32)

                # mask = tf.expand_dims(mask, axis=0)
                # print(m)
                full_states = self.lstm_layers(inputs=states)
                ind = tf.reduce_sum(mask,axis=-1)
                states = tf.gather_nd(full_states, tf.transpose([
                    tf.range(mask.get_shape()[0]),ind
                ]))
                # states = full_states[:,ind]

            
            for layer in self.encode_layers:
                states = layer(states)
            
            # if self.bptt:
            #     return states,[h_last,cell_last]
        
        return states

    def call(self, states,mask, test=False):

        #encode part
        # for layer in self.encode_layers:
        #     states = layer(states)
        # print(mask.get_shape())
        states = self._state_encoding(states,mask,test)
        
        actor_feature = tf.stop_gradient(states)
        critic_feature = states
        #actor part
        dist = self._compute_dist(actor_feature,test=test)
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

        #critic part
        for layer in self.critic_layers:
            critic_feature = layer(critic_feature)
        values = tf.squeeze(critic_feature, axis=1)

        return actions, log_pis,values
    def compute_log_probs(self, states, actions,mask):
        raw_actions = actions / self._max_action
        # for layer in self.encode_layers:
        #     states = layer(states)
        states = self._state_encoding(states,mask=mask)
        states = tf.stop_gradient(states)
        dist = self._compute_dist(states)
        logp_pis = dist.log_prob(raw_actions)
        return logp_pis

    def compute_entropy(self, states,mask):
        # for layer in self.encode_layers:
        #     states = layer(states)
        states = self._state_encoding(states,mask=mask)
        states = tf.stop_gradient(states)
        dist = self._compute_dist(states)
        return dist.entropy()
