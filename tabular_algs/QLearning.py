import numpy as np
from load_balance_gym.envs.param import config

class QLearning:
    def __init__(self, env, num_states, num_actions, alpha, gamma, epsilon):
        self.env = env
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def run_q_learning(self, num_episodes, verbose=True):
        self.init_agent()
        rewards_per_episode = np.array([None] * num_episodes)
        episode_len = np.array([None] * num_episodes)

        for episode in range(num_episodes):
            print('MCControl: episode ', episode)
            state_action_reward = self.generate_episode(self.policy)
            G = self.calculate_returns(state_action_reward)
            self.evaluate_policy(G)
            self.improve_policy()

            total_return = 0
            for _, _, reward in state_action_reward:
                total_return += reward
            rewards_per_episode[episode] = total_return
            episode_len = len(state_action_reward)

        final_policy = self.argmax(self.Q, self.policy)

        if verbose:
            print(f"Finished training RL agent for {num_episodes} episodes!")

        return self.Q, final_policy, rewards_per_episode, episode_len

    def init_agent(self):
        self.policy = np.random.choice(self.num_actions, self.num_states)
        self.Q = {}

        for state in range(self.num_states):
            self.Q[state] = {}
            for action in range(self.num_actions):
                self.Q[state][action] = 0

    def tuple_to_num(self, s):
        #state = 0
        #for pos in range(len(s)):
            #state = state * config.load_balance_queue_size + s[pos]
        #return state
        for pos in range(len(s)):
            if s[pos] >= config.load_balance_queue_size:
                return config.load_balance_queue_size-1
        return (s[0]//100)*config.load_balance_queue_size**2 + s[1]*config.load_balance_queue_size + s[2]

    def generate_episode(self, policy):
        G = 0
        s = self.env.reset()
        state = self.tuple_to_num(s)
        a = policy[state]
        state_action_reward = [(state, a, 0)]
        while True:
            s, r, terminated, _ = self.env.step(a)
            if terminated:
                state_action_reward.append((state, None, r))
                break
            else:
                a = policy[state]
                state_action_reward.append((state, a, r))
        return state_action_reward

    def calculate_returns(self, state_action_reward):
        G = {}
        t = 0
        for state, action, reward in state_action_reward:
            if state not in G:
                G[state] = {action: 0}
            else:
                if action not in G[state]:
                    G[state][action] = 0
            for s in G.keys():
                for a in G[s].keys():
                    G[s][a] += reward * self.gamma ** t
            t += 1
        return G

    def evaluate_policy(self, G):
        for state in G.keys():
            for action in G[state].keys():
                if action == self.policy[state]:
                    self.visit_count[state][action] += 1
                    self.Q[state][action] += 1/self.visit_count[state][action]*(G[state][action]-self.Q[state][action])
        return self.Q

    def improve_policy(self):
        self.policy = self.argmax(self.Q, self.policy)
        for state in range(self.num_states):
            self.policy[state] = self.get_epsilon_greedy_action(self.policy[state])
        return self.policy

    def argmax(self, Q, policy):
        next_policy = policy
        for state in range(self.num_states):
            best_action = None
            best_value = float('-inf')
            for action in range(self.num_actions):
                if Q[state][action] > best_value:
                    best_action = action
                    best_value = self.Q[state][action]
            next_policy[state] = best_action
        return next_policy

    def get_epsilon_greedy_action(self, greedy_action):
        prob = np.random.random()
        if prob < 1 - self.epsilon:
            next_action = greedy_action
        else:
            next_action = np.random.choice(self.num_actions, 1)
        return next_action