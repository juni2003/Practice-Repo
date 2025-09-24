# Deep Q-Networks (DQN)

Deep Q-Networks (DQN) are a type of Reinforcement Learning algorithm that combines Q-Learning with deep neural networks. This allows the agent to learn in environments with high-dimensional state spaces, such as image-based environments like video games.

## The Problem with Tabular Q-Learning

Traditional Q-Learning uses a table (the Q-table) to store the Q-values for every state-action pair. This works well for environments with a small number of states and actions. However, in many real-world problems, the state space can be enormous or even continuous.

For example, in a video game where the state is the raw pixel data of the screen, the number of possible states is astronomically large. It's impossible to create a Q-table to store a value for every possible screen configuration.

## The DQN Solution

DQN solves this problem by using a neural network to approximate the Q-function. Instead of a Q-table, we have a **Q-network** that takes the state as input and outputs the Q-values for all possible actions in that state.

`Q(s, a; θ) ≈ Q*(s, a)`

Here, `θ` represents the weights of the neural network. The network learns to approximate the optimal Q-function by minimizing a loss function, typically the Mean Squared Error (MSE) between the predicted Q-values and the target Q-values.

## Key Innovations in DQN

The original DQN paper by DeepMind introduced two key techniques to stabilize the training process:

### 1. Experience Replay

Instead of training the network on consecutive samples, which can be highly correlated, DQN uses **experience replay**. The agent's experiences `(state, action, reward, next_state)` are stored in a replay buffer (a memory). During training, mini-batches of experiences are randomly sampled from this buffer.

This has several advantages:
-   It breaks the correlation between consecutive samples, making the training more stable.
-   It allows the agent to learn from the same experience multiple times, increasing data efficiency.

### 2. Target Network

In Q-Learning, the same network is used to both predict the Q-value and to calculate the target Q-value. This can lead to instabilities because the target is constantly changing.

DQN uses a separate **target network** to calculate the target Q-values. The target network is a copy of the Q-network, but its weights are updated much less frequently (e.g., every few thousand steps). This provides a more stable target for the Q-network to learn from.

The loss function for DQN is:

`L(θ) = E[(R + γ * max_a' Q(s', a'; θ') - Q(s, a; θ))^2]`

-   `θ`: Weights of the Q-network.
-   `θ'`: Weights of the target network.
