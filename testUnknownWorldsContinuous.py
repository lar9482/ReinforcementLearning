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
import openpyxl

class testParameter_UnknownWorldCont:
    def __init__(self, worldName, world, discount, maxExpectedReward, maxNumTries, radius):
        self.worldName = worldName
        self.world = world
        self.discount = discount
        self.maxExpectedReward = maxExpectedReward
        self.maxNumTries = maxNumTries
        self.radius = radius
        self.numEpisodes = 500

def getContinuousDataset():
    worldOptions = {
        'world1Cont': getWorld1Continuous(),
        'world2Cont': getWorld2Continuous(),
        'world3Cont': getWorld3Continuous(),
    }
    discountOptions = [0.1, 0.5, 0.9]
    maxExpectedRewardOptions = [1, 2.5, 5]
    maxNumTriesOptions = [10, 50, 100]
    radiusOptions = [1, 2.5, 5]

    dataset = {
        'world1Cont': [],
        'world2Cont': [],
        'world3Cont': [],
    }
    for worldName in list(worldOptions.keys()):
        world = worldOptions[worldName]
        for discount in discountOptions:
            for maxExpectedReward in maxExpectedRewardOptions:
                for maxNumTries in maxNumTriesOptions:
                    for radius in radiusOptions:
                        parameter = testParameter_UnknownWorldCont(
                            worldName,
                            world,
                            discount,
                            maxExpectedReward,
                            maxNumTries,
                            radius
                        )

                        dataset[worldName].append(parameter)
    
    return dataset

def runUnknownWorldTest_Continuous(testParameter_UnknownWorld, lock):
    world = testParameter_UnknownWorld.world
    QLearnAgent = functionQAgent(
        world,
        learnType.QLearn,
        testParameter_UnknownWorld.discount,
        testParameter_UnknownWorld.maxExpectedReward,
        testParameter_UnknownWorld.maxNumTries,
        0,
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
        0.50,
        testParameter_UnknownWorld.radius
    )
    SARSA_Epsilon_75Percent = functionQAgent(
        world,
        learnType.SASRA,
        testParameter_UnknownWorld.discount,
        testParameter_UnknownWorld.maxExpectedReward,
        testParameter_UnknownWorld.maxNumTries,
        0.75,
        testParameter_UnknownWorld.radius
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

    lock.acquire()
    saveMeanStdOfAvgRewardPerEpisode(
        testParameter_UnknownWorld,
        avgRewardPerEpisode_Qlearn,
        avgRewardPerEpisode_SARSA_Epsilon_25Percent,
        avgRewardPerEpisode_SARSA_Epsilon_50Percent,
        avgRewardPerEpisode_SARSA_Epsilon_75Percent     
    )
    lock.release()

def runAgent(agent, world, numEpisodes):
    avgRewardPerEpisode = []
    for episode in range(0, numEpisodes):
        state = world.initial_state
        action = random.choice(list(world.actions))
        totalRuns = 0
        allRewards = []
        while (not world.is_terminal(state) and totalRuns < 500):
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

def generateHeatMap(agent, row, column, fileName):
    
    dataNorth = np.zeros((row, column), dtype=float)
    dataEast = np.zeros((row, column), dtype=float)
    dataSouth = np.zeros((row, column), dtype=float)
    dataWest = np.zeros((row, column), dtype=float)

    for x in range(column):
        for y in range(row):
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
    createHeatMap(column, row, [dataSouth, dataEast, dataNorth, dataWest], filePath)

def saveMeanStdOfAvgRewardPerEpisode(
        testParameter_UnknownWorld,
        avgRewardPerEpisode_Qlearn,
        avgRewardPerEpisode_SARSA_Epsilon_25Percent,
        avgRewardPerEpisode_SARSA_Epsilon_50Percent,
        avgRewardPerEpisode_SARSA_Epsilon_75Percent     
    ):

    meanQLearn = sum(avgRewardPerEpisode_Qlearn) / len(avgRewardPerEpisode_Qlearn)
    varianceQLearn = sum([((x - meanQLearn) ** 2) for x in avgRewardPerEpisode_Qlearn]) / len(avgRewardPerEpisode_Qlearn)
    stdQLearn = varianceQLearn  ** 0.5

    meanSARSA25 = sum(avgRewardPerEpisode_SARSA_Epsilon_25Percent) / len(avgRewardPerEpisode_SARSA_Epsilon_25Percent)
    varianceSARSA25 = sum([((x - meanSARSA25) ** 2) for x in avgRewardPerEpisode_SARSA_Epsilon_25Percent]) / len(avgRewardPerEpisode_SARSA_Epsilon_25Percent)
    stdSARSA25 = varianceSARSA25 ** 0.5

    meanSARSA50 = sum(avgRewardPerEpisode_SARSA_Epsilon_50Percent) / len(avgRewardPerEpisode_SARSA_Epsilon_50Percent)
    varianceSARSA50 = sum([((x - meanSARSA50) ** 2) for x in avgRewardPerEpisode_SARSA_Epsilon_50Percent]) / len(avgRewardPerEpisode_SARSA_Epsilon_50Percent)
    stdSARSA50 = varianceSARSA50 ** 0.5

    meanSARSA75 = sum(avgRewardPerEpisode_SARSA_Epsilon_75Percent) / len(avgRewardPerEpisode_SARSA_Epsilon_75Percent)
    varianceSARSA75 = sum([((x - meanSARSA75) ** 2) for x in avgRewardPerEpisode_SARSA_Epsilon_75Percent]) / len(avgRewardPerEpisode_SARSA_Epsilon_75Percent)
    stdSARSA75 = varianceSARSA75 ** 0.5

    statWorkbook = openpyxl.load_workbook('./results/WorldUnknown/MeansAndSTDCont.xlsx')
    statSheet = statWorkbook.active
    
    statSheet.append([
        testParameter_UnknownWorld.worldName,
        testParameter_UnknownWorld.discount,
        testParameter_UnknownWorld.maxExpectedReward,
        testParameter_UnknownWorld.maxNumTries,
        testParameter_UnknownWorld.radius,
        "{:.3f}".format(meanQLearn),
        "{:.3f}".format(stdQLearn),
        "{:.3f}".format(meanSARSA25),
        "{:.3f}".format(stdSARSA25),
        "{:.3f}".format(meanSARSA50),
        "{:.3f}".format(stdSARSA50),
        "{:.3f}".format(meanSARSA75),
        "{:.3f}".format(stdSARSA75)
    ])

    statWorkbook.save('./results/WorldUnknown/MeansAndSTDCont.xlsx')
    statWorkbook.close()

def getFileName(testParameter_UnknownWorld, epsilon):
    return '{0}-ε_{1}-discount_{2}-maxExpectedReward_{3}-maxNumTries_{4}-radius_{5}'.format(
        str(testParameter_UnknownWorld.worldName),
        str(epsilon),
        str(testParameter_UnknownWorld.discount),
        str(testParameter_UnknownWorld.maxExpectedReward),
        str(testParameter_UnknownWorld.maxNumTries),
        str(testParameter_UnknownWorld.radius),
    )