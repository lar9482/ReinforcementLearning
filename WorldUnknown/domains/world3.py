from MDP.gridworld_c import ContinuousGridWorldMDP

def getWorld3Continuous():
    mdp = ContinuousGridWorldMDP(10, 10)
    mdp.add_goal(5, 5, 1, 1)

    return mdp