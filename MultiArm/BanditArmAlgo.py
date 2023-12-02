from MultiArm.bandit_sim import Bandit_Sim

import sys

class BanditArmAlgo:
    """
        The bandit algorithm will keep track of the actual reward means
    """
    def __init__(self, numArms, payoutSTD):
        self.banditSim = Bandit_Sim(numArms, payoutSTD)
        self.numArms = numArms
        self.actualRewardMeansPerArm = {arm: 0 for arm in range(0, self.numArms)}
        self.numPullsPerArm = {arm: 0 for arm in range(0, self.numArms)}

    def pullAllArms(self):
        V = 0
        # Trying all arms before the repeat loop.
        for arm in range(0, self.numArms):
            reward = self.banditSim.pull_arm(arm)

            V += reward
            self.actualRewardMeansPerArm[arm] = (
                (self.numPullsPerArm[arm] * self.actualRewardMeansPerArm[arm] + reward) /
                (self.numPullsPerArm[arm] + 1)
            )
            self.numPullsPerArm[arm] += 1
        
        return V

    def getExpectedRewardMean(self, arm):
        return self.banditSim.arm_means[arm]
    
    def getActualRewardMean(self, arm):
        return self.actualRewardMeansPerArm[arm]
    
    def getExpectedRegret(self, arm):
        rewardStar = max(self.banditSim.arm_means)
        return rewardStar - self.banditSim.arm_means[arm]

    def getActualRegret(self, arm):
        rewardStar = -sys.maxsize
        for armPrime in range(0, self.numArms):
            rewardStar = max(rewardStar, self.actualRewardMeansPerArm[armPrime])
        
        return rewardStar - self.actualRewardMeansPerArm[arm]

    def runAlgorithm(self, numSamples):
        pass