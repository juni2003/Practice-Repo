# Markov Decision Processes (MDPs)

A Markov Decision Process (MDP) is a mathematical framework for modeling decision-making in situations where outcomes are partly random and partly under the control of a decision-maker. MDPs are the foundation of most Reinforcement Learning algorithms.

An MDP is defined by a tuple `(S, A, P, R, γ)`:

-   **S:** A set of states.
-   **A:** A set of actions.
-   **P:** The state transition probability function, `P(s' | s, a)`. This is the probability of transitioning to state `s'` from state `s` after taking action `a`.
-   **R:** The reward function, `R(s, a, s')`. This is the immediate reward received after transitioning from state `s` to state `s'` due to action `a`.
-   **γ (gamma):** The discount factor, where `0 ≤ γ ≤ 1`. It determines the importance of future rewards. A value of 0 means the agent is short-sighted and only cares about the immediate reward. A value close to 1 means the agent is far-sighted and values future rewards highly.

## The Markov Property

The "Markov" in MDP comes from the **Markov Property**. This property states that the future is independent of the past, given the present. In other words, the current state `s` provides enough information to make an optimal decision, and the history of how the agent arrived at that state is not relevant.

`P(St+1 | St) = P(St+1 | S1, S2, ..., St)`

This simplifies the problem significantly, as the agent only needs to consider the current state to take the best action.

## Solving MDPs

The goal in an MDP is to find a policy `π(s)` that maximizes the cumulative discounted reward. This is known as finding the **optimal policy**, denoted as `π*`.

The optimal policy can be found using dynamic programming methods like:

-   **Value Iteration:** This algorithm iteratively updates the state-value function until it converges to the optimal value function `V*`.
-   **Policy Iteration:** This algorithm alternates between two steps:
    1.  **Policy Evaluation:** Calculate the value function for the current policy.
    2.  **Policy Improvement:** Improve the policy by acting greedily with respect to the value function.

While these methods can find the optimal policy, they require a complete model of the environment (i.e., the transition probabilities and rewards), which is often not available in real-world RL problems. This is where model-free methods like Q-Learning come in.
