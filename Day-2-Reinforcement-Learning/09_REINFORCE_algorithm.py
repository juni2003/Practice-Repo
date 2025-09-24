import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym

# Define the Policy Network
class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

# REINFORCE Agent
class ReinforceAgent:
    def __init__(self, state_size, action_size):
        self.policy = Policy(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.gamma = 0.99
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def update_policy(self):
        R = 0
        policy_loss = []
        returns = []

        # Calculate discounted rewards
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        # Normalize returns for more stable training
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        # Clear memory
        del self.rewards[:]
        del self.saved_log_probs[:]

# Main training loop
if __name__ == "__main__":
    # Using OpenAI Gym's CartPole environment
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = ReinforceAgent(state_size, action_size)
    episodes = 1000

    for e in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            state, reward, done, _, _ = env.step(action)
            agent.rewards.append(reward)
            episode_reward += reward

        agent.update_policy()

        if e % 50 == 0:
            print(f"Episode {e}\tLast reward: {episode_reward:.2f}")

    env.close()
