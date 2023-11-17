from testKnownWorlds import testKnownWorlds
from testUnknownWorldsDiscrete import testUnknownWorldsQLearnDiscrete, testUnknownWorldsSARSADiscrete
from testUnknownWorldsContinuous import testUnknownWorldsQLearnContinuous

def main():
    # testKnownWorlds()
    # testUnknownWorldsQLearnDiscrete()
    # testUnknownWorldsSARSADiscrete()
    testUnknownWorldsQLearnContinuous()
    
if __name__ == '__main__':
    main()