import gym
import load_balance_gym

env = gym.make('load_balance-v1', )

obs = env.reset()
n_steps = 10

print('env.action_space', env.action_space)

for step in range(n_steps):
    print('step = ', step)

    action = env.action_space.sample()
    print('action = ', action)

    state, reward, done, info = env.step(action)
    print('state = ', state)
    print('reward = ', reward)