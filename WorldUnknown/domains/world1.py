from MDP.gridworld_c import ContinuousGridWorldMDP
from MDP.gridworld import DiscreteGridWorldMDP

def getWorld1Continuous():
    mdp = ContinuousGridWorldMDP(4, 3)

    # mdp.add_pit(1, 1, 1, 0)
    mdp.add_pit(4, 2, 1, -1)
    mdp.add_goal(4, 3, 1, 1)

    return mdp

def getWorld1Discrete():
    mdp = DiscreteGridWorldMDP(4, 3, -0.01)

    mdp.add_obstacle('pit', [4, 2], -1)
    # mdp.add_obstacle('pit', [1, 1], 0)
    mdp.add_obstacle('goal', [4, 3], 1)

    return mdp