import numpy as np

# Define the environment
# A simple 1D environment with 6 states (0-5)
# The goal is to reach state 5.
# The agent can move left or right.

# R-matrix (Rewards)
# -1 represents no direct path.
# 0 represents a possible move.
# 100 is the reward for reaching the goal.
R = np.matrix([[-1, -1, -1, -1, 0, -1],
               [-1, -1, -1, 0, -1, 100],
               [-1, -1, -1, 0, -1, -1],
               [-1, 0, 0, -1, 0, -1],
               [0, -1, -1, 0, -1, 100],
               [-1, 0, -1, -1, 0, 100]])

# Q-matrix (Quality)
# Initialize Q-matrix with all zeros.
Q = np.zeros_like(R, dtype=float)

# Hyperparameters
gamma = 0.8  # Discount factor
alpha = 0.8  # Learning rate
epsilon = 0.1 # Epsilon for epsilon-greedy strategy

# Training the agent
for episode in range(1000):
    # Start from a random state
    state = np.random.randint(0, 6)

    while state != 5:  # While not in the goal state
        # Get all possible actions for the current state
        possible_actions = np.where(R[state, :] >= 0)[1]

        # Epsilon-greedy strategy
        if np.random.uniform(0, 1) < epsilon:
            # Exploration: choose a random action
            action = np.random.choice(possible_actions)
        else:
            # Exploitation: choose the action with the highest Q-value
            # If all Q-values are the same, choose a random action
            if np.all(Q[state, possible_actions] == Q[state, possible_actions][0]):
                 action = np.random.choice(possible_actions)
            else:
                action = possible_actions[np.argmax(Q[state, possible_actions])]


        # Get the next state
        next_state = action

        # Update the Q-value for the current state and action
        # Q(s, a) = Q(s, a) + alpha * [R(s, a) + gamma * max(Q(s', a')) - Q(s, a)]
        q_sa = Q[state, action]
        max_q_s_prime = np.max(Q[next_state, :])
        reward = R[state, action]

        Q[state, action] = q_sa + alpha * (reward + gamma * max_q_s_prime - q_sa)

        # Move to the next state
        state = next_state

print("Trained Q-matrix:")
print(Q.astype(int))

# Testing the agent
print("\nTesting the trained agent:")
# Start from state 2
current_state = 2
steps = [current_state]

while current_state != 5:
    next_step_index = np.where(Q[current_state, :] == np.max(Q[current_state, :]))[1]

    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size=1))
    else:
        next_step_index = int(next_step_index)

    steps.append(next_step_index)
    current_state = next_step_index

print("Most efficient path from state 2 to 5:")
print(steps)
