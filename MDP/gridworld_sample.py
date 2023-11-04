from MDP.gridworld import DiscreteGridWorldMDP
import numpy as np

def gridworldSample():
    mdp = DiscreteGridWorldMDP(50, 50)

    mdp.add_obstacle('pit', [5, 5])
    mdp.add_obstacle('pit', [5, 10])
    mdp.add_obstacle('pit', [10, 20])
    mdp.add_obstacle('pit', [40, 41])

    mdp.add_obstacle('goal', [50, 50])

    mdp.display()

    x = mdp.initial_state
    t = 0

    while not mdp.is_terminal(x) and t < 1000:
        print(x)
        a = np.random.choice(list(mdp.actions_at(x)))
        print(a)
        xPrime, prob = list(zip(*mdp.p(x, a)))
        x, _ = mdp.act(x, a)
        t += 1

    print(x)