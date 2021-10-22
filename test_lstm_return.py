from tensorflow.keras.layers import LSTM,Dense
import tensorflow as tf
layers = LSTM(256,return_sequences=True,return_state=True)
l = Dense(16)
a = tf.random.uniform((32,3,16))
c = tf.random.uniform((32,3,3))
# from tf2rl.algos.ppo import PPO
# from actor_critic_policy import GaussianActorCritic
# import numpy as np
# state_input = True
# LSTM = True
# N_steps = 3
# policy = PPO(
#     state_shape=(3,16),
#     action_dim=2,
#     is_discrete=False,
#     state_input=state_input,
#     lstm=LSTM,
#     batch_size=32,
#     horizon=512,
#     n_epoch=4,
#     lr_actor=5e-4,
#     trans=False,
#     use_schdule=False,
#     final_steps=10000,
#     final_lr=1e-5,
#     entropy_coef=0.01, 
#     vfunc_coef=0.5,
#     actor_critic=GaussianActorCritic(
#         state_shape=(3,16),
#         action_dim=2,
#         max_action=1.,
#         squash=True,
#         state_input=state_input,
#         lstm=LSTM,
#         trans=False
#     )
# )

# batch=32
# data = np.load(
#     '/home/haochen/TPDM_transformer/test.npz'
# )
# for _ in range(10):
#     policy.train(
#         states=data['states'],
#         actions=data['actions'],
#         advantages=data['advantages'],
#         logp_olds=data['logp_olds'],
#         returns=data['returns']
#     )

b1 = layers(a)
b2 = layers(c)
print(b1.get_shape())
print(b2.get_shape())
#,bs.get_shape())
# for i in b:
#     print(i)
#     print(i.get_shape())