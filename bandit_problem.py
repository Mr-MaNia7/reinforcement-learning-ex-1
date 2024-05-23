import numpy as np

class MultiArmedBandit:
    def __init__(self, k):
        self.k = k
        self.action_values = np.random.normal(0, 1, k)  # True values of actions
        self.estimates = np.zeros(k)  # Estimated values
        self.action_count = np.zeros(k)

    def pull(self, action):
        reward = np.random.normal(self.action_values[action], 1)  # Reward is normally distributed around the true value
        self.update_estimates(action, reward)
        return reward

    def update_estimates(self, action, reward):
        self.action_count[action] += 1
        alpha = 1 / self.action_count[action]
        self.estimates[action] += alpha * (reward - self.estimates[action])

    def reset(self):
        self.estimates = np.zeros(self.k)
        self.action_count = np.zeros(self.k)

def epsilon_greedy(bandit, episodes, epsilon):
    rewards = []
    for _ in range(episodes):
        if np.random.random() < epsilon:
            action = np.random.choice(bandit.k)
        else:
            action = np.argmax(bandit.estimates)
        reward = bandit.pull(action)
        rewards.append(reward)
    return np.sum(rewards)

def ucb(bandit, episodes, c):
    rewards = []
    for t in range(1, episodes + 1):
        ucb_values = bandit.estimates + c * np.sqrt(np.log(t) / (bandit.action_count + 1e-5))
        action = np.argmax(ucb_values)
        reward = bandit.pull(action)
        rewards.append(reward)
    return np.sum(rewards)

def thompson_sampling(bandit, episodes):
    rewards = []
    alpha = np.ones(bandit.k)
    beta = np.ones(bandit.k)

    for _ in range(episodes):
        samples = [np.random.beta(alpha[i], beta[i]) if alpha[i] > 0 and beta[i] > 0 else 0 for i in range(bandit.k)]
        action = np.argmax(samples)
        reward = bandit.pull(action)
        
        # Assume reward is already binary (0 or 1)
        alpha[action] += reward
        beta[action] += 1 - reward
        
        rewards.append(reward)
    return np.sum(rewards)


def value_iteration(bandit, iterations):
    for _ in range(iterations):
        for a in range(bandit.k):
            bandit.estimates[a] = np.max([bandit.pull(a) for _ in range(10)])  # Simplistic approximation
    return bandit.estimates

def policy_iteration(bandit, iterations):
    policy = np.random.randint(bandit.k, size=bandit.k)
    for _ in range(iterations):
        bandit.reset()
        # Policy Evaluation
        for _ in range(100):
            for a in range(bandit.k):
                bandit.pull(policy[a])
        # Policy Improvement
        for a in range(bandit.k):
            q_values = [bandit.pull(a) for _ in range(10)]
            policy[a] = np.argmax(q_values)
    return policy

# Simulation setup
k = 10  # Number of arms
episodes = 1000  # Number of pulls
bandit = MultiArmedBandit(k)

# Run and evaluate the algorithms
print("Epsilon-Greedy Total Reward:", epsilon_greedy(bandit, episodes, 0.1))
bandit.reset()
print("UCB Total Reward:", ucb(bandit, episodes, 2))
bandit.reset()
print("Thompson Sampling Total Reward:", thompson_sampling(bandit, episodes))
bandit.reset()
print("Value Iteration Estimates:", value_iteration(bandit, 10))
bandit.reset()
print("Policy Iteration Policy:", policy_iteration(bandit, 10))
