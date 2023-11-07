from MDP.gridworld_c import ContinuousGridWorldMDP

def getWorld2Continuous():
    mdp = ContinuousGridWorldMDP(10, 10)
    mdp.add_goal(10, 10, 1, 1)

    return mdp