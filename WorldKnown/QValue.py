from MDP.hashStates import hashState

"""
    Calculates the Q value given the current state and action
    @param mdp: MDP
        The MDP simulator
    
    @param discount: Float
        The discount factor
    
    @param currState: MDPState
        The current state
    
    @param currAction: Actions
        The current action
    
    @param U: {hashState(state): float}
        The current utility function
        NOTE: 
        The key is a hashed version of 'state' using 'hashState'
    
    @returns float:
        The QValue itself
"""
def QValue(mdp, discount, currState, currAction, U):
    nextStateAndProbs = {
        nextState: transProb 
        for nextState, transProb in mdp.p(currState, currAction)
    }

    totalU = 0
    for nextState in list(nextStateAndProbs.keys()):
        transProb = nextStateAndProbs[nextState]
        reward = mdp.r(currState, nextState)
        
        hashedNextState = hashState(nextState)
        if (U.get(hashedNextState) != None):
            totalU += (transProb) * (reward + discount*U[hashedNextState])

    return totalU