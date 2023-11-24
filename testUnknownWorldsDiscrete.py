from WorldUnknown.domains.world1 import getWorld1Discrete
from WorldUnknown.domains.world2 import getWorld2Discrete
from WorldUnknown.domains.world3 import getWorld3Discrete
from WorldUnknown.QAgent import learnType
from WorldUnknown.tableQAgent import tableQAgent

from MDP.hashStates import hashState
from MDP.gridworld import Actions, GridState
from heatmap import createHeatMap

import random
import numpy as np
import matplotlib.pyplot as plt

class testParameter_KnownWorld:
    def __init__(self, worldName, world, discount, maxExpectedReward, maxNumTries):
        self.worldName = worldName
        self.world = world
        self.discount = discount
        self.maxExpectedReward = maxExpectedReward
        self.maxNumTries = maxNumTries
        self.numEpisodes = 1000

def getDataset():
    worldOptions = {
        'world1Dis': getWorld1Discrete(),
        'world2Dis': getWorld2Discrete(),
        'world3Dis': getWorld3Discrete(),
    }
    discountOptions = [0.01, 0.1, 0.5, 0.9]
    maxExpectedRewardOptions = [1, 2.5, 5]
    maxNumTriesOptions = [10, 50]

    dataset = {
        'world1Dis': [],
        'world2Dis': [],
        'world3Dis': [],
    }
    for worldName in list(worldOptions.keys()):
        world = worldOptions[worldName]
        for discount in discountOptions:
            for maxExpectedReward in maxExpectedRewardOptions:
                for maxNumTries in maxNumTriesOptions:
                    parameter = testParameter_KnownWorld(
                        worldName,
                        world,
                        discount,
                        maxExpectedReward,
                        maxNumTries
                    )

                    dataset[worldName].append(parameter)
    
    return dataset

def runKnownWorldTest(testParameter_UnknownWorld):
    world = testParameter_UnknownWorld.world
    QLearnAgent = tableQAgent(
        world,
        learnType.QLearn,
        testParameter_UnknownWorld.discount,
        testParameter_UnknownWorld.maxExpectedReward,
        testParameter_UnknownWorld.maxNumTries,
        0,
    )

    SARSA_Epsilon_25Percent = tableQAgent(
        world,
        learnType.SASRA,
        testParameter_UnknownWorld.discount,
        testParameter_UnknownWorld.maxExpectedReward,
        testParameter_UnknownWorld.maxNumTries,
        0.25,
    )
    SARSA_Epsilon_50Percent = tableQAgent(
        world,
        learnType.SASRA,
        testParameter_UnknownWorld.discount,
        testParameter_UnknownWorld.maxExpectedReward,
        testParameter_UnknownWorld.maxNumTries,
        0.50,
    )
    SARSA_Epsilon_75Percent = tableQAgent(
        world,
        learnType.SASRA,
        testParameter_UnknownWorld.discount,
        testParameter_UnknownWorld.maxExpectedReward,
        testParameter_UnknownWorld.maxNumTries,
        0.75,
    )

    avgRewardPerEpisode_Qlearn = runAgent(QLearnAgent, world, testParameter_UnknownWorld.numEpisodes)
    avgRewardPerEpisode_SARSA_Epsilon_25Percent = runAgent(SARSA_Epsilon_25Percent, world, testParameter_UnknownWorld.numEpisodes)
    avgRewardPerEpisode_SARSA_Epsilon_50Percent = runAgent(SARSA_Epsilon_50Percent, world, testParameter_UnknownWorld.numEpisodes)
    avgRewardPerEpisode_SARSA_Epsilon_75Percent = runAgent(SARSA_Epsilon_75Percent, world, testParameter_UnknownWorld.numEpisodes)

    generateHeatMap(QLearnAgent, world.height, world.width, 
        getFileName(testParameter_UnknownWorld, QLearnAgent.epsilon)
    )
    generateHeatMap(SARSA_Epsilon_25Percent, world.height, world.width, 
        getFileName(testParameter_UnknownWorld, 0.25)       
    )
    generateHeatMap(SARSA_Epsilon_50Percent, world.height, world.width, 
        getFileName(testParameter_UnknownWorld, 0.50)       
    )
    generateHeatMap(SARSA_Epsilon_75Percent, world.height, world.width, 
        getFileName(testParameter_UnknownWorld, 0.75)       
    )

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
            
        avgRewardPerEpisode.append(sum(allRewards) / len(allRewards))
        rewardLog = '{0}:{1}-{2}'.format(
            str(episode+1),
            str(agent.learnType),
            str(sum(allRewards) / len(allRewards))
        )
        print(rewardLog)
    return avgRewardPerEpisode

def generateHeatMap(agent, row, column, fileName):
    
    dataNorth = np.zeros((row+1, column+1), dtype=float)
    dataEast = np.zeros((row+1, column+1), dtype=float)
    dataSouth = np.zeros((row+1, column+1), dtype=float)
    dataWest = np.zeros((row+1, column+1), dtype=float)

    for x in range(column+1):
        for y in range(row+1):
            # agent.Q[(hashState(state), action)]
            dataNorth[y][x] = agent.Q[
                hashState(GridState(x, y, column, row)), Actions.UP
            ]
            dataEast[y][x] = agent.Q[
                hashState(GridState(x, y, column, row)), Actions.RIGHT
            ]
            dataSouth[y][x] = agent.Q[
                hashState(GridState(x, y, column, row)), Actions.DOWN
            ]
            dataWest[y][x] = agent.Q[
                hashState(GridState(x, y, column, row)), Actions.LEFT
            ]
    
    dataNorth = (dataNorth - np.min(dataNorth)) / (np.max(dataNorth) - np.min(dataNorth))
    dataEast = (dataEast - np.min(dataEast)) / (np.max(dataEast) - np.min(dataEast))
    dataSouth = (dataSouth - np.min(dataSouth)) / (np.max(dataSouth) - np.min(dataSouth))
    dataWest = (dataWest - np.min(dataWest)) / (np.max(dataWest) - np.min(dataWest))
    filePath = './results/WorldUnknown/{0}-{1}.png'.format(
        str(agent.learnType),
        fileName
    )
    createHeatMap(column+1, row+1, [dataNorth, dataEast, dataSouth, dataWest], filePath)

def plotAvgRewardPerEpisode_QLearn(
        testParameter_UnknownWorld, 
        avgRewardPerEpisode_Qlearn,
        avgRewardPerEpisode_SARSA_Epsilon_25Percent,
        avgRewardPerEpisode_SARSA_Epsilon_50Percent,
        avgRewardPerEpisode_SARSA_Epsilon_75Percent
    ):

    fileName = getFileName(testParameter_UnknownWorld, '0')
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

def getFileName(testParameter_UnknownWorld, epsilon):
    return '{0}-ε_{1}-discount_{2}-maxExpectedReward_{3}-maxNumTries_{4}-radius_{5}'.format(
        str(testParameter_UnknownWorld.worldName),
        str(epsilon),
        str(testParameter_UnknownWorld.discount),
        str(testParameter_UnknownWorld.maxExpectedReward),
        str(testParameter_UnknownWorld.maxNumTries),
        str(testParameter_UnknownWorld.radius),
    )