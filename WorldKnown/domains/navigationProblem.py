from MDP.gridworld import DiscreteGridWorldMDP

def getNavigationProblem():
    mdp = DiscreteGridWorldMDP(4, 3, -0.04)

    mdp.add_obstacle('pit', [3, 1], -1)
    mdp.add_obstacle('pit', [1, 1], 0)
    mdp.add_obstacle('goal', [3, 2], 1)
    
    return mdp