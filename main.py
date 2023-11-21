from testKnownWorlds import testKnownWorlds
from testUnknownWorldsDiscrete import testUnknownWorldsQLearnDiscrete, testUnknownWorldsSARSADiscrete
from testUnknownWorldsContinuous import testUnknownWorldsQLearnContinuous, testUnknownWorldsSARSAContinuous

def main():
    # testKnownWorlds()
    # testUnknownWorldsQLearnDiscrete()
    # testUnknownWorldsSARSADiscrete()
    # testUnknownWorldsQLearnContinuous()
    testUnknownWorldsSARSAContinuous()
    
if __name__ == '__main__':
    main()