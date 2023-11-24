from WorldUnknown.domains.world1 import getWorld1Continuous, getWorld1Discrete
from WorldUnknown.domains.world2 import getWorld2Continuous, getWorld2Discrete
from WorldUnknown.domains.world3 import getWorld3Continuous, getWorld3Discrete

from WorldUnknown.QAgent import learnType
from WorldUnknown.functionQAgent import functionQAgent

from MDP.hashStates import hashState
from MDP.gridworld_c import Actions
from heatmap import createHeatMap

import random
import numpy as np
import matplotlib.pyplot as plt

class testParameter_UnknownWorld:
    def __init__(self, worldName, world, discount, maxExpectedReward, maxNumTries, radius):
        self.worldName = worldName
        self.world = world
        self.discount = discount
        self.maxExpectedReward = maxExpectedReward
        self.maxNumTries = maxNumTries
        self.radius = radius
        self.numEpisodes = 1000

def getDataset():
    worldOptions = {
        'world1': getWorld1Continuous(),
        'world2': getWorld2Continuous(),
        'world3': getWorld3Continuous(),
    }
    discountOptions = [0.01, 0.1, 0.5, 0.9]
    maxExpectedRewardOptions = [1, 2.5, 5]
    maxNumTriesOptions = [10, 50]
    radiusOptions = [1, 1.5, 3]

    dataset = {
        'world1': [],
        'world2': [],
        'world3': [],
    }
    for worldName in list(worldOptions.keys()):
        world = worldOptions[worldName]
        for discount in discountOptions:
            for maxExpectedReward in maxExpectedRewardOptions:
                for maxNumTries in maxNumTriesOptions:
                    for radius in radiusOptions:
                        parameter = testParameter_UnknownWorld(
                            worldName,
                            world,
                            discount,
                            maxExpectedReward,
                            maxNumTries,
                            radius
                        )

                        dataset[worldName].append(parameter)
    
    return dataset

def runUnknownWorldTest(testParameter_UnknownWorld):
    world = testParameter_UnknownWorld.world
    QLearnAgent = functionQAgent(
        world,
        learnType.QLearn,
        testParameter_UnknownWorld.discount,
        testParameter_UnknownWorld.maxExpectedReward,
        testParameter_UnknownWorld.maxNumTries,
        -0.01,
        testParameter_UnknownWorld.radius
    )

    SARSA_Epsilon_25Percent = functionQAgent(
        world,
        learnType.SASRA,
        testParameter_UnknownWorld.discount,
        testParameter_UnknownWorld.maxExpectedReward,
        testParameter_UnknownWorld.maxNumTries,
        0.25,
        testParameter_UnknownWorld.radius
    )
    SARSA_Epsilon_50Percent = functionQAgent(
        world,
        learnType.SASRA,
        testParameter_UnknownWorld.discount,
        testParameter_UnknownWorld.maxExpectedReward,
        testParameter_UnknownWorld.maxNumTries,
        0.25,
        testParameter_UnknownWorld.radius
    )
    SARSA_Epsilon_75Percent = functionQAgent(
        world,
        learnType.SASRA,
        testParameter_UnknownWorld.discount,
        testParameter_UnknownWorld.maxExpectedReward,
        testParameter_UnknownWorld.maxNumTries,
        0.25,
        testParameter_UnknownWorld.radius
    )

    avgRewardPerEpisode_Qlearn = runAgent(QLearnAgent, world, testParameter_UnknownWorld.numEpisodes)
    avgRewardPerEpisode_SARSA_Epsilon_25Percent = runAgent(SARSA_Epsilon_25Percent, world, testParameter_UnknownWorld.numEpisodes)
    avgRewardPerEpisode_SARSA_Epsilon_50Percent = runAgent(SARSA_Epsilon_50Percent, world, testParameter_UnknownWorld.numEpisodes)
    avgRewardPerEpisode_SARSA_Epsilon_75Percent = runAgent(SARSA_Epsilon_75Percent, world, testParameter_UnknownWorld.numEpisodes)

    fileName = getFileName(testParameter_UnknownWorld)
    generateHeatMap(QLearnAgent, world.height, world.width, fileName)
    generateHeatMap(SARSA_Epsilon_25Percent, world.height, world.width, fileName)
    generateHeatMap(SARSA_Epsilon_50Percent, world.height, world.width, fileName)
    generateHeatMap(SARSA_Epsilon_75Percent, world.height, world.width, fileName)

    plotAvgRewardPerEpisode_QLearn(
        testParameter_UnknownWorld, 
        avgRewardPerEpisode_Qlearn,
        avgRewardPerEpisode_SARSA_Epsilon_25Percent,
        avgRewardPerEpisode_SARSA_Epsilon_50Percent,
        avgRewardPerEpisode_SARSA_Epsilon_75Percent
    )

def runAgent(agent, world, numEpisodes):
    avgRewardPerEpisode = []
    for episode in range(0, numEpisodes):
        state = world.initial_state
        action = random.choice(list(world.actions))
        totalRuns = 0
        allRewards = []
        while (not world.is_terminal(state) and totalRuns < 10000):
            state, r = world.act(state, action)
            action = agent.learn(state, r)
            allRewards.append(r)
            print()
        avgRewardPerEpisode.append(sum(allRewards) / len(allRewards))

    return avgRewardPerEpisode

def plotAvgRewardPerEpisode_QLearn(
        testParameter_UnknownWorld, 
        avgRewardPerEpisode_Qlearn,
        avgRewardPerEpisode_SARSA_Epsilon_25Percent,
        avgRewardPerEpisode_SARSA_Epsilon_50Percent,
        avgRewardPerEpisode_SARSA_Epsilon_75Percent
    ):

    fileName = getFileName(testParameter_UnknownWorld)
    filePath = './results/WorldUnknown/{0}.png'.format(fileName)

    plt.xlim(0, len(avgRewardPerEpisode_Qlearn))
    plt.ylim(
        min(
        *avgRewardPerEpisode_Qlearn, 
        *avgRewardPerEpisode_SARSA_Epsilon_25Percent, 
        *avgRewardPerEpisode_SARSA_Epsilon_50Percent, 
        *avgRewardPerEpisode_SARSA_Epsilon_75Percent) - 0.2, 
         max(
        *avgRewardPerEpisode_Qlearn, 
        *avgRewardPerEpisode_SARSA_Epsilon_25Percent, 
        *avgRewardPerEpisode_SARSA_Epsilon_50Percent, 
        *avgRewardPerEpisode_SARSA_Epsilon_75Percent) + 0.2
    )
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward') 
    plt.title('Average Rewards per Episode')
    plt.plot([i for i in range(0, len(avgRewardPerEpisode_Qlearn))], avgRewardPerEpisode_Qlearn, label='QLearn')
    plt.plot(
        [i for i in range(0, len(avgRewardPerEpisode_SARSA_Epsilon_25Percent))], 
        avgRewardPerEpisode_SARSA_Epsilon_25Percent, label='SARSA_ε0.25'
    )
    plt.plot(
        [i for i in range(0, len(avgRewardPerEpisode_SARSA_Epsilon_25Percent))], 
        avgRewardPerEpisode_SARSA_Epsilon_50Percent, label='SARSA_ε0.50'
    )
    plt.plot(
        [i for i in range(0, len(avgRewardPerEpisode_SARSA_Epsilon_25Percent))], 
        avgRewardPerEpisode_SARSA_Epsilon_75Percent, label='SARSA_ε0.75'
    )
    plt.legend(loc='upper right')
    plt.savefig(filePath)
    plt.clf()

def generateHeatMap(agent, row, column, fileName):
    
    dataNorth = np.zeros((row+1, column+1), dtype=float)
    dataEast = np.zeros((row+1, column+1), dtype=float)
    dataSouth = np.zeros((row+1, column+1), dtype=float)
    dataWest = np.zeros((row+1, column+1), dtype=float)

    for x in range(column+1):
        for y in range(row+1):
            dataNorth[y][x] = agent.calculateQValue(
                np.array([y, x]), Actions.UP
            )
            dataEast[y][x] = agent.calculateQValue(
                np.array([y, x]), Actions.RIGHT
            )
            dataSouth[y][x] = agent.calculateQValue(
                np.array([y, x]), Actions.DOWN
            )
            dataWest[y][x] = agent.calculateQValue(
                np.array([y, x]), Actions.LEFT
            )
    
    dataNorth = (dataNorth - np.min(dataNorth)) / (np.max(dataNorth) - np.min(dataNorth))
    dataEast = (dataEast - np.min(dataEast)) / (np.max(dataEast) - np.min(dataEast))
    dataSouth = (dataSouth - np.min(dataSouth)) / (np.max(dataSouth) - np.min(dataSouth))
    dataWest = (dataWest - np.min(dataWest)) / (np.max(dataWest) - np.min(dataWest))
    filePath = './results/WorldUnknown/{0}-{1}.png'.format(
        str(agent.learnType),
        fileName
    )
    createHeatMap(column+1, row+1, [dataNorth, dataEast, dataSouth, dataWest], filePath)

def getFileName(testParameter_UnknownWorld):
    return '{0}-discount_{1}-maxExpectedReward_{2}-maxNumTries_{3}-radius_{4}'.format(
        str(testParameter_UnknownWorld.worldName),
        str(testParameter_UnknownWorld.discount),
        str(testParameter_UnknownWorld.maxExpectedReward),
        str(testParameter_UnknownWorld.maxNumTries),
        str(testParameter_UnknownWorld.radius)
    )

def testUnknownWorldsQLearnContinuous():
    world = getWorld3Continuous()
    
    maxExpectedReward = 1
    maxNumTries = 10
    discount = 0.1
    epsilon = 0.2
    radius = 1

    typeAgent = learnType.QLearn
    agent = functionQAgent(world, typeAgent, discount, maxExpectedReward, maxNumTries, epsilon, radius)

    convergeCount = 0
    for _ in range(0, 10):

        state = world.initial_state
        action = random.choice(list(world.actions))
        totalR = 0
        totalRuns = 0

        while (not world.is_terminal(state) and totalRuns < 250):
            state, r = world.act(state, action)
            action = agent.learn(state, r)
            totalR += r
            totalRuns += 1

        if (convergeCount >= 5):
            break

        if ((totalR / totalRuns) == -0.009999999999999875):
            convergeCount += 1
        else:
            convergeCount = 0

        print(totalR / totalRuns)
    
    generateHeatMap(agent, 10, 10)
    print('#####################')
    test1 = np.array([4, 4])
    test2 = np.array([6, 6])
    print(agent.calculateQValue(test1, Actions.UP))
    print(agent.calculateQValue(test1, Actions.DOWN))
    print(agent.calculateQValue(test1, Actions.LEFT))
    print(agent.calculateQValue(test1, Actions.RIGHT))
    print('#####################')
    print(agent.calculateQValue(test2, Actions.UP))
    print(agent.calculateQValue(test2, Actions.DOWN))
    print(agent.calculateQValue(test2, Actions.LEFT))
    print(agent.calculateQValue(test2, Actions.RIGHT))
    print()

    # test1 = GridState(4, 4, 10, 10)
    # test2 = GridState(6, 6, 10, 10)

    # print(agent.calculateQValue(test1, Actions.UP))
    # print(agent.calculateQValue(test1, Actions.DOWN))
    # print(agent.calculateQValue(test1, Actions.LEFT))
    # print(agent.calculateQValue(test1, Actions.RIGHT))
    # print('#####################')
    # print(agent.calculateQValue(test2, Actions.UP))
    # print(agent.calculateQValue(test2, Actions.DOWN))
    # print(agent.calculateQValue(test2, Actions.LEFT))
    # print(agent.calculateQValue(test2, Actions.RIGHT))
    # print()

def testUnknownWorldsSARSAContinuous():
    world = getWorld3Continuous()
    
    maxExpectedReward = 10
    maxNumTries = 50
    discount = 0.1
    epsilon = 0.3
    radius = 1

    typeAgent = learnType.SASRA
    agent = functionQAgent(world, typeAgent, discount, maxExpectedReward, maxNumTries, epsilon, radius)

    convergeCount = 0
    for _ in range(0, 50):

        state = world.initial_state
        action = random.choice(list(world.actions))
        totalR = 0
        totalRuns = 0

        while (not world.is_terminal(state) and totalRuns < 5000):
            state, r = world.act(state, action)
            action = agent.learn(state, r)
            totalR += r
            totalRuns += 1

        if (convergeCount >= 5):
            break

        if ((totalR / totalRuns) == -0.09):
            convergeCount += 1
        else:
            convergeCount = 0

        print(totalR / totalRuns)
    
    print('#####################')
    test1 = np.array([1, 1])
    print(agent.calculateQValue(test1, Actions.UP))
    print(agent.calculateQValue(test1, Actions.DOWN))
    print(agent.calculateQValue(test1, Actions.LEFT))
    print(agent.calculateQValue(test1, Actions.RIGHT))
    print('#####################')
    test2 = np.array([3, 3])
    print(agent.calculateQValue(test2, Actions.UP))
    print(agent.calculateQValue(test2, Actions.DOWN))
    print(agent.calculateQValue(test2, Actions.LEFT))
    print(agent.calculateQValue(test2, Actions.RIGHT))
    print('#####################')
    test3 = np.array([7, 7])
    print(agent.calculateQValue(test3, Actions.UP))
    print(agent.calculateQValue(test3, Actions.DOWN))
    print(agent.calculateQValue(test3, Actions.LEFT))
    print(agent.calculateQValue(test3, Actions.RIGHT))
    print('#####################')
    test4 = np.array([9, 9])
    print(agent.calculateQValue(test4, Actions.UP))
    print(agent.calculateQValue(test4, Actions.DOWN))
    print(agent.calculateQValue(test4, Actions.LEFT))
    print(agent.calculateQValue(test4, Actions.RIGHT))

    # test1 = GridState(3, 3, 10, 10)
    # test2 = GridState(7, 7, 10, 10)

    # print(agent.calculateQValue(test1, Actions.UP))
    # print(agent.calculateQValue(test1, Actions.DOWN))
    # print(agent.calculateQValue(test1, Actions.LEFT))
    # print(agent.calculateQValue(test1, Actions.RIGHT))
    # print('#####################')
    # print(agent.calculateQValue(test2, Actions.UP))
    # print(agent.calculateQValue(test2, Actions.DOWN))
    # print(agent.calculateQValue(test2, Actions.LEFT))
    # print(agent.calculateQValue(test2, Actions.RIGHT))
    # print()