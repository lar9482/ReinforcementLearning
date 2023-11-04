from MDP.hashStates import hashState

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