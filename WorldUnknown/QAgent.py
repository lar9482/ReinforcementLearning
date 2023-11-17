from enum import Enum

class learnType(Enum):
    QLearn = 0
    SASRA = 1

class QAgent:
    def learn(self, currState, currReward):
        """
            The main source where the agent will learn
        """
        pass

    def QLearn(self, state, reward):
        """
            Implementation of QLearning with some form of

            Q[s, a] ← Q[s, a] + α(Nsa[s, a])(r + γ maxa′ Q[s′, a′] − Q[s, a])
        """
        pass

    def SARSA(self, state, reward):
        """
            Implementation of SARSA with some form of 

            Q(s, a) ← Q(s, a) + α[R(s, a, s′) + γ Q(s′, a′) − Q(s, a)]
        """
        pass
    
    def explore(self, U, N, maxNumTries, maxExpectedReward):
        if (N < maxNumTries):
            return maxExpectedReward
        else:
            return U

    def learningRate(self, N):
        return 1 / N
