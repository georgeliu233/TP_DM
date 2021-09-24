import os
import gym
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES']='3'

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec,Agent

import sys
#sys.path.append('/home/haochen/SMARTS_test_TPDM/sac_model/sac_pic.py')
from tf2rl.experiments.on_policy_trainer import OnPolicyTrainer
from tf2rl.algos.ppo import PPO
agent_spec = AgentSpec(
    interface=AgentInterface.from_type(AgentType.LanerWithSpeed, max_episode_steps=1000,neighborhood_vehicles=True),
    agent_builder=None
)
agent_specs={
    'Agent-LHC':agent_spec
}
env = gym.make(
    "smarts.env:hiway-v0",
    scenarios=["scenarios/left_turn"],
    agent_specs=agent_specs,
)

LSTM = True
N_steps = 3

env.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
# env.observation_space = gym.spaces.Box(low=0,high=255,shape=(80,80,3), dtype=np.float32)
env.observation_space = gym.spaces.Box(low=-10000,high=10000,shape=(5,16), dtype=np.float32)

parser = OnPolicyTrainer.get_argument()
parser = PPO.get_argument(parser)
args = parser.parse_args()

args.max_steps=100000
#args.model_dir='/home/haochen/SMARTS_test_TPDM/sac_model/tf2rl_model'
#args.normalize_obs=True
args.logdir='/home/haochen/SMARTS_test_TPDM/sac_model/tf2rl_ppo'
args.test_episodes=10
args.save_summary_interval=int(1e2)
state_input = True

policy = PPO(
    state_shape=env.observation_space.shape,
    action_dim=2,
    is_discrete=False,
    state_input=True,
    lstm=False,
    batch_size=32,
    horizon=1024,
    n_epoch=1,
    trans=True
)

trainer = OnPolicyTrainer(policy,env,args,test_env=env,state_input=True,lstm=True,n_steps=5)
trainer()
