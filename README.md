# 4 project from CS 5313-01

Advanced Artificial Intelligence

Taught by Dr. Sandip Sen at the University of Tulsa in FA 23.

## This repository implements and experiments with numerous Reinforcement Learning Algorithms: 

#### Algorithms for solving small and discrete Markov Decision Processes:
  - Value Iteration for solving utility functions, *U(s)* for every state s.
    - $`U(s) = max_{a \in A(s)} \sum_{s'} P(s'|s,a) (R(s,a,s') + \gamma U(s'))`$
  - Policy Iteration for solving the policy $`\pi^{*}(s)`$, for every state s.
    - $`\pi^{*}(s) = argmax_{a \in A(s)} \sum_{s'} P(s'|s,a) (R(s,a,s') + \gamma U(s'))`$

#### Approximate Solving techniques using Q-Learning and its variants, and different agent architectures
  - Q-Learning:
    - $`Q(s, a) = Q(s, a) + \alpha(r + \gamma max_{a'}Q(s', a') - Q(s, a))`$
  - SARSA:
    - $`Q(s, a) = Q(s, a) + \alpha(r + \gamma Q(s', a') - Q(s, a))`$, where $`a'`$ is chosen with a $`\epsilon`$-greedy policy.
  - Agent Architectures:
    - Tabular Representation:
      - Storing Q(s, a) as an indexable table
    - Function Approximation representation
      - Calculating $`Q(s, a) = \theta_1 + \theta_2*X + \theta_3*Y`$
