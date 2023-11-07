from MDP.gridworld_c import ContinuousGridWorldMDP

def getWorld1Continuous():
    mdp = ContinuousGridWorldMDP(4, 3)

    mdp.add_pit(1, 1, 1, 0)
    mdp.add_pit(4, 2, 1, -1)
    mdp.add_goal(4, 3, 1, 1)

    return mdp