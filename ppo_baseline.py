import os
import gym
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES']='2'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec,Agent

import sys
#sys.path.append('/home/haochen/SMARTS_test_TPDM/sac_model/sac_pic.py')
from on_policy_trainer import OnPolicyTrainer
from tf2rl.algos.ppo import PPO
from actor_critic_policy import GaussianActorCritic
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
    headless=True
)


state_input = True
LSTM = True
N_steps = 8
neighbours = 5
use_neighbors=True

env.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

if state_input:
    if use_neighbors:
        env.observation_space = gym.spaces.Box(low=-10000,high=10000,shape=(N_steps,4*(neighbours+1)), dtype=np.float32)
    else:
        env.observation_space = gym.spaces.Box(low=-10000,high=10000,shape=(N_steps,16), dtype=np.float32)
else:
    env.observation_space = gym.spaces.Box(low=0,high=255,shape=(80,80,3), dtype=np.float32)

parser = OnPolicyTrainer.get_argument()
parser = PPO.get_argument(parser)
args = parser.parse_args()

args.max_steps=200000
#args.model_dir='/home/haochen/SMARTS_test_TPDM/sac_model/tf2rl_model'
#args.normalize_obs=True
args.logdir='/home/haochen/TPDM_transformer/ppo_log'
args.test_episodes=20
args.test_interval=2500
args.save_summary_interval=int(1e2)
args.normalize_obs=False


policy = PPO(
    state_shape=env.observation_space.shape,
    action_dim=2,
    is_discrete=False,
    state_input=state_input,
    lstm=LSTM,
    batch_size=32,
    horizon=512,
    n_epoch=8,
    lr_actor=5e-4,
    trans=False,
    use_schdule=False,
    final_steps=args.max_steps,
    final_lr=1e-5,
    entropy_coef=0.01, 
    vfunc_coef=0.5,
    gpu=-1,
    actor_critic=GaussianActorCritic(
        state_shape=env.observation_space.shape,
        action_dim=2,
        max_action=1.,
        squash=True,
        state_input=state_input,
        lstm=bool(LSTM&(~use_neighbors)),
        trans=False,
        ego_surr=use_neighbors,use_trans=True,neighbours=neighbours,time_step=N_steps
    )
)

print(use_neighbors)
trainer = OnPolicyTrainer(policy=policy,env=env,args=args,test_env=env,state_input=state_input,lstm=LSTM,n_steps=N_steps,
ego_surr=use_neighbors,surr_vehicles=neighbours,save_name='ppo_neighbor')
trainer()
