from MDP.gridworld_c import ContinuousGridWorldMDP
from MDP.gridworld import DiscreteGridWorldMDP

def getWorld3Continuous():
    mdp = ContinuousGridWorldMDP(10, 10)
    mdp.add_goal(5, 5, 1, 10)

    return mdp

def getWorld3Discrete():
    mdp = DiscreteGridWorldMDP(10, 10, -0.01)
    mdp.add_obstacle('goal', [5, 5], 1)

    return mdp