from MultiArm.UCB import UCB
from MultiArm.EGreedy import EGreedy
def testMultiArm():
    numArms = 10
    numSamples = 10
    payoutSTD = 0.25

    UCBInstance = UCB(numArms, payoutSTD)
    cumRewardUCB = UCBInstance.runAlgorithm(numSamples)

    epsilon = 0.25
    EGreedyInstance = EGreedy(numArms, payoutSTD, epsilon)
    cumRewardEGreedy = EGreedyInstance.runAlgorithm(numSamples)

    print(cumRewardUCB)
    print(cumRewardEGreedy)
