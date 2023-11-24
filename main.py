from testKnownWorlds import testKnownWorlds
from testUnknownWorldsDiscrete import testUnknownWorldsQLearnDiscrete, testUnknownWorldsSARSADiscrete
from testUnknownWorldsContinuous import testUnknownWorldsQLearnContinuous, testUnknownWorldsSARSAContinuous
from testMultiArm import testMultiArm

from testUnknownWorldsContinuous import runUnknownWorldTest, getDataset, testParameter_UnknownWorld
from WorldUnknown.domains.world3 import getWorld3Continuous
from WorldUnknown.domains.world1 import getWorld1Continuous
def testUnknownWorlds():
    dataset = getDataset()
    parameter = testParameter_UnknownWorld(
        'world1',
        getWorld1Continuous(),
        0.1,
        1,
        10,
        1
    )
    runUnknownWorldTest(parameter)
    
def main():
    # testKnownWorlds()
    # testUnknownWorldsQLearnDiscrete()
    # testUnknownWorldsSARSADiscrete()
    # testUnknownWorldsQLearnContinuous()
    # testUnknownWorldsSARSAContinuous()
    testUnknownWorlds()
    # testMultiArm()
    
if __name__ == '__main__':
    main()