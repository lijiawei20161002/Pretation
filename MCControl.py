class MCControl:
    def __init__(self, env, num_states, num_actions, epsilon, gamma):
        self.env = env
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.gamma = gamma

    def run_mc_control(self, num_episodes, verbose=True):
        self.init_agent()
        rewards_per_episode = np.array([None] * num_episodes)
        episode_len = np.array([None] * num_episodes)

        for episode in range(num_episodes):
            state_action_reward = self.generate_episode(self.policy)
            G = self.calculate_returns(state_action_reward)
            self.evaluate_policy(G)
            self.improve_policy()

            total_return = 0
            for _, _, reward in state_action_reward:
                total_return += reward
            rewards_per_episode[episode] = total_return
            episode_len = len(state_action_reward)

        final_policy = self.argmax(seelf.Q, self.policy)

        if verbose:
            print(f"Finished training RL agent for {num_episodes} episodes!")

        return self.Q, final_policy, rewards_per_episode, episode_len

    def init_agent(self):
        self.policy = np.random.choice(num_actions, num_states)
        self.Q = {}
        self.visit_count = {}

        for state in range(self.num_states):
            self.Q[state] = {}
            self.visit_count[state] = {}
            for action in range(self.num_actions):
                self.Q[state][action] = 0
                self.visit_count[state][action] = 0

    def generate_episode(self, policy):
        G = 0
        s = env.reset()
        a = policy[s]

        state_action_reward = [(s, a, 0)]
        while True:
            s, r, terminated, _ = env.step(a)
            if terminated:
                state_action_reward.append((s, None, r))
                break
            else:
                a = policy[s]
                state_action_reward.append((s, a, r))
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
                    G[s][a] += reward * gamma ** t
            t += 1
        return G

    def evaluate_policy(self, G):
        for state in G.keys():
            for action in G[state].keys():
                if action:
                    self.visit_count[state][action] += 1
                    self.Q[state][action] += 1/self.visit_count[state][action]*(G[state][action]-self.Q[state][action])
        return self.Q

    def improve_policy(self):
        self.policy = argmax(Q, self.policy)
        for state in range(self.num_state):
            self.policy[state] = get_epsilon_greedy_action(self.policy[state])
        return self.policy

    def argmax(self, Q, policy):
        next_policy = policy
        for state in range(self.num_states):
            best_action = None
            best_value = float('-inf')
            for action in range(self.num_actions):
                if self.Q[state][action] > best_value:
                    best_action = action
            next_policy[state] = best_action
        return next_policy

    def get_epsilon_greedy_action(self, greedy_action):
        prob = np.random.random()
        if prob < 1 - self.epsilon:
            next_action = greedy_action
        else:
            next_action = np.random.choice(self.num_action, 1)
        return next_action