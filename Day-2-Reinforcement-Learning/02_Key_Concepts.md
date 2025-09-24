# Key Concepts in Reinforcement Learning

To understand Reinforcement Learning, it's essential to be familiar with its core components:

## 1. Agent

The **Agent** is the learner or decision-maker. It perceives the environment, takes actions, and receives rewards. In a game, the agent could be the player. In a self-driving car, it could be the control system.

## 2. Environment

The **Environment** is everything the agent interacts with. It represents the world in which the agent exists and operates. The environment takes the agent's current state and action as input and returns the agent's reward and next state as output.

## 3. State

A **State** is a complete description of the environment at a particular point in time. It's a snapshot of the world that the agent needs to make a decision. For example, in a chess game, the state would be the positions of all the pieces on the board.

## 4. Action

An **Action** is a move the agent can make in the environment. The set of all possible actions in a given state is called the **action space**.

## 5. Reward

A **Reward** is the feedback the agent receives from the environment after taking an action in a particular state. The reward signal is a scalar value that tells the agent how good or bad its action was. The agent's goal is to maximize the cumulative reward.

## 6. Policy

The **Policy** is the agent's strategy. It's a mapping from states to actions. A policy defines the agent's behavior at a given time. It can be:
-   **Deterministic:** For a given state, the policy always returns the same action.
-   **Stochastic:** For a given state, the policy returns a probability distribution over actions.

## 7. Value Function

The **Value Function** estimates how good it is for the agent to be in a particular state (or to take a particular action in a state). It's a prediction of the expected future reward. There are two main types:
-   **State-Value Function (V):** The expected return starting from a state `s` and then following a policy `π`.
-   **Action-Value Function (Q):** The expected return starting from a state `s`, taking an action `a`, and then following a policy `π`. This is often called the "Q-function".
