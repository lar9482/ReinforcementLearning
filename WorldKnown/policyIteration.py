import random
import sys
from WorldKnown.QValue import QValue
from MDP.hashStates import hashState

def policyIteration(mdp, discount, k):
    U = {hashState(state): 0 for state in list(mdp.states)}
    policy = {
        hashState(state): random.choice(mdp.actions_at(state))
        for state in list(mdp.states)
    }

    while True:
        policyEvaluation(U, policy, mdp, discount, k)
        unchanged = True
        
        for currState in list(mdp.states):
            argMaxAction = argmaxQValue(mdp, discount, currState, U)
            QValueArgMaxAction = QValue(mdp, discount, currState, argMaxAction, U)

            policyQValue = QValue(mdp, discount, currState, policy[hashState(currState)], U)
            if (QValueArgMaxAction > policyQValue):
                policy[hashState(currState)] = argMaxAction
                unchanged = False
        
        if (unchanged):
            break

    return policy

def policyEvaluation(U, policy, mdp, discount, k):
    for _ in range(0, k):
        for currState in list(mdp.states):
            
            updatedU = 0
            actionFromPolicy = policy[hashState(currState)]
            nextStateAndProbs = {
                nextState: transProb 
                for nextState, transProb in mdp.p(currState, actionFromPolicy)
            }
            
            for nextState in list(nextStateAndProbs.keys()):
                if (U.get(hashState(nextState)) == None):
                    continue

                transProbFromCurrToNet = nextStateAndProbs[nextState]
                updatedU += (
                    (transProbFromCurrToNet)*(mdp.r(currState, nextState) + discount*U[hashState(nextState)])
                )
            
            U[hashState(currState)] = updatedU

def argmaxQValue(mdp, discount, currState, U):
    actionsAtCurrState = mdp.actions_at(currState)
    maxQValue = -sys.maxsize
    argmaxAction = actionsAtCurrState[0]

    for currAction in actionsAtCurrState:
        currQValue = QValue(mdp, discount, currState, currAction, U)
        
        if (maxQValue < currQValue):
            maxQValue = currQValue
            argmaxAction = currAction
    
    return argmaxAction