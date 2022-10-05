import gym
import load_balance_gym
from stable_baselines.deepq.policies import MlpPolicy as mlp_dqn
from stable_baselines.common.policies import MlpPolicy as mlp
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines.sac.policies import MlpPolicy as mlp_sac
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN
from stable_baselines import A2C
from stable_baselines import ACER
from stable_baselines import ACKTR
from stable_baselines import DQN
from stable_baselines import GAIL
from stable_baselines import PPO1
from stable_baselines import PPO2
from stable_baselines import TRPO
# without DDPG, TD3, SAC only intended to work with continuous actions, not support Discrete action space
# without HER, HER requires the environment to inherits from gym.GoalEnv

import matplotlib.pyplot as plt
import numpy as np

env = gym.make('load_balance-v1', )

model_class = DQN
goal_selection_strategy = 'future'

# add GAIL later
algs = ['A2C', 'ACER', 'ACKTR', 'DQN', 'PPO1', 'PPO2', 'TRPO']
models = ['model_A2C = A2C(mlp, env, verbose=1, tensorboard_log="./log/")',
'model_ACER = ACER(mlp, env, verbose=1, tensorboard_log="./log/")',
'model_ACKTR = ACKTR(mlp, env, verbose=1, tensorboard_log="./log/")',
'model_DQN = DQN(mlp_dqn, env, verbose=1, tensorboard_log="./log/")',
'model_PPO1 = PPO1(mlp, env, verbose=1, tensorboard_log="./log/")',
'model_PPO2 = PPO2(mlp, env, verbose=1, tensorboard_log="./log/")',
'model_TRPO = TRPO(mlp, env, verbose=1, tensorboard_log="./log/")'
]
trains = ['model_A2C.learn(total_timesteps=25000)',
'model_ACER.learn(total_timesteps=25000)',
'model_ACKTR.learn(total_timesteps=25000)',
'model_DQN.learn(total_timesteps=25000)',
'model_PPO1.learn(total_timesteps=25000)',
'model_PPO2.learn(total_timesteps=25000)',
'model_TRPO.learn(total_timesteps=25000)'
]

print('env.action_space', env.action_space)
#plt.yscale('log')

for i in range(len(algs)):
    alg = algs[i]
    print(alg)
    obs = env.reset()
    n_steps = 100
    rewards = []
    exec(models[i])
    exec(trains[i])
    obs = env.reset()
    for step in range(n_steps):
        exec('action, _states = model_'+alg+'.predict(obs)')
        state, reward, done, info = env.step(action)
        rewards.append(reward)
    plt.plot(rewards, label=alg)

plt.legend()
plt.savefig('robust.png')