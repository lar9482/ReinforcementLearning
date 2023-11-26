from testKnownWorlds import testKnownWorlds
from testMultiArm import testMultiArm

from testUnknownWorldsContinuous import runUnknownWorldTest_Continuous, getContinuousDataset
from testUnknownWorldsDiscrete import runUnKnownWorldTest_Discrete, getDiscreteDataset
from WorldUnknown.domains.world3 import getWorld3Continuous
from WorldUnknown.domains.world1 import getWorld1Continuous, getWorld1Discrete

from multiprocessing import Process, Manager

def testUnknownWorlds_Discrete():
    dataset = getDiscreteDataset()
    for worldName in list(dataset.keys()):
        parameterList = dataset[worldName]

        with Manager() as manager:
            allProcesses = []
            lock = manager.Lock()

            for parameter in parameterList:

                process = Process(
                    target=runUnKnownWorldTest_Discrete, 
                    args=(
                        parameter,
                        lock
                    )
                )
                allProcesses.append(process)
            
            for process in allProcesses:
                process.start()

            for process in allProcesses:
                process.join()

def testUnknownWorlds_Continuous():
    dataset = getContinuousDataset()
    for worldName in list(dataset.keys()):
        parameterList = dataset[worldName]

        with Manager() as manager:
            allProcesses = []
            lock = manager.Lock()

            for parameter in parameterList:

                process = Process(
                    target=runUnknownWorldTest_Continuous, 
                    args=(
                        parameter,
                        lock
                    )
                )
                allProcesses.append(process)
            
            for process in allProcesses:
                process.start()

            for process in allProcesses:
                process.join()
    
def main():
    # testKnownWorlds()
    # testUnknownWorldsQLearnDiscrete()
    # testUnknownWorldsSARSADiscrete()
    # testUnknownWorldsQLearnContinuous()
    # testUnknownWorldsSARSAContinuous()
    # testUnknownWorlds_Discrete()
    # testUnknownWorlds_Continuous()
    testMultiArm()
    
if __name__ == '__main__':
    main()