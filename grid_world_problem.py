import gymnasium as gym
import numpy as np
import random
import math

# Initialize the environment
env = gym.make('FrozenLake-v1', is_slippery=False)

# Hyperparameters
gamma = 0.99  # Discount factor
theta = 1e-6  # Convergence threshold
alpha = 0.1  # Learning rate for Q-Learning
epsilon = 0.1  # Exploration rate for Q-Learning
episodes = 10000  # Number of episodes for Q-Learning
N = 10000  # Total number of steps for UCB
c = 1  # Exploration parameter for UCB

# Value Iteration
def value_iteration(env, gamma, theta):
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = V[s]
            max_value = float('-inf')
            for a in range(env.action_space.n):
                q = sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]])
                if q > max_value:
                    max_value = q
            V[s] = max_value
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

def derive_policy_from_values(env, V, gamma):
    policy = np.zeros(env.observation_space.n, dtype=int)
    for s in range(env.observation_space.n):
        max_value = float('-inf')
        best_action = 0
        for a in range(env.action_space.n):
            q = sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]])
            if q > max_value:
                max_value = q
                best_action = a
        policy[s] = best_action
    return policy

# Policy Iteration
def policy_evaluation(policy, env, gamma, theta):
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = V[s]
            action = policy[s]
            V[s] = sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][action]])
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

def policy_iteration(env, gamma, theta):
    policy = np.zeros(env.observation_space.n, dtype=int)
    while True:
        V = policy_evaluation(policy, env, gamma, theta)
        policy_stable = True
        for s in range(env.observation_space.n):
            old_action = policy[s]
            action_values = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                action_values[a] = sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]])
            new_action = np.argmax(action_values)
            if old_action != new_action:
                policy[s] = new_action
                policy_stable = False
        if policy_stable:
            break
    return policy, V

# Q-Learning with Epsilon-Greedy
def q_learning(env, alpha, gamma, epsilon, episodes):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            # Check if state is a tuple and extract the integer if so
            if isinstance(state, tuple):
                state = state[0]  # Assuming the state is the first element of the tuple

            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(Q[state])  # Exploit

            # Handling potentially extra values returned by env.step
            results = env.step(action)
            next_state, reward, done = results[:3]  # Unpack the first three elements

            # Check if next_state is a tuple and extract the integer if so
            if isinstance(next_state, tuple):
                next_state = next_state[0]

            best_next_action = np.argmax(Q[next_state])
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])
            state = next_state
    return Q


# Upper Confidence Bound (UCB)
def ucb_algorithm(env, N, c):
    counts = np.zeros(env.action_space.n)
    values = np.zeros(env.action_space.n)
    total_reward = 0
    
    for t in range(1, N + 1):
        # Compute the UCB values for each action, balancing exploration and exploitation
        ucb_values = values + c * np.sqrt(np.log(t) / (counts + 1e-5))
        action = np.argmax(ucb_values)
        
        # Perform a step in the environment with the selected action
        results = env.step(action)
        
        # Unpack results while handling potentially extra return values gracefully
        next_state, reward, done = results[:3]
        
        # Update counts and estimated values for the selected action
        counts[action] += 1
        # Update the estimated value using incremental averaging to improve numerical stability
        values[action] += (reward - values[action]) / counts[action]
        
        total_reward += reward
        
        # Optionally, handle early termination if the episode ends
        if done:
            env.reset()  # Reset the environment for a new episode

    return values

# Run Value Iteration
optimal_values_vi = value_iteration(env, gamma, theta)
optimal_policy_vi = derive_policy_from_values(env, optimal_values_vi, gamma)
print("Value Iteration - Optimal State-Value Function:")
print(optimal_values_vi)
print("Value Iteration - Derived Policy:")
print(optimal_policy_vi)

# Run Policy Iteration
optimal_policy_pi, optimal_values_pi = policy_iteration(env, gamma, theta)
print("Policy Iteration - Optimal Policy:")
print(optimal_policy_pi)
print("Policy Iteration - Optimal State-Value Function:")
print(optimal_values_pi)

# Run Q-Learning
optimal_Q = q_learning(env, alpha, gamma, epsilon, episodes)
print("Q-Learning - Optimal Q-Table:")
print(optimal_Q)

# Run UCB
optimal_values_ucb = ucb_algorithm(env, N, c)
print("UCB - Optimal Values using UCB:")
print(optimal_values_ucb)
