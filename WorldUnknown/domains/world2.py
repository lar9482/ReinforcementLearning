from MDP.gridworld_c import ContinuousGridWorldMDP
from MDP.gridworld import DiscreteGridWorldMDP

def getWorld2Continuous():
    mdp = ContinuousGridWorldMDP(10, 10)
    mdp.add_goal(10, 10, 1, 1)

    return mdp

def getWorld2Discrete():
    mdp = DiscreteGridWorldMDP(10, 10, -0.01)
    mdp.add_obstacle('goal', [10, 10], 1)

    return mdp