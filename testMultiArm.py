from MultiArm.UCB import UCB

def testMultiArm():
    numArms = 10
    numSamples = 10
    payoutSTD = 0.25

    UCBInstance = UCB(numArms, payoutSTD)
    V = UCBInstance.runAlgorithm(numSamples)
    print(V)
