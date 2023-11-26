from MultiArm.bandit_sim import Bandit_Sim

import sys

class BanditArmAlgo:
    def __init__(self, numArms, payoutSTD):
        self.banditSim = Bandit_Sim(numArms, payoutSTD)
        self.numArms = numArms
        self.rewardMeansPerArm = {arm: 0 for arm in range(0, self.numArms)}
        self.numPullsPerArm = {arm: 0 for arm in range(0, self.numArms)}

    def pullAllArms(self):
        V = 0
        # Trying all arms before the repeat loop.
        for arm in range(0, self.numArms):
            reward = self.banditSim.pull_arm(arm)

            V += reward
            self.rewardMeansPerArm[arm] = (
                (self.numPullsPerArm[arm] * self.rewardMeansPerArm[arm] + reward) /
                (self.numPullsPerArm[arm] + 1)
            )
            self.numPullsPerArm[arm] += 1
        
        return V

    def expectedRegret(self, arm):
        maxRewardMean = -sys.maxsize
        for possibleArm in list(range(self.numArms)):
            maxRewardMean = max(maxRewardMean, self.rewardMeansPerArm[possibleArm])
        
        return maxRewardMean - self.rewardMeansPerArm[arm]
    
    def runAlgorithm(self, numSamples):
        pass