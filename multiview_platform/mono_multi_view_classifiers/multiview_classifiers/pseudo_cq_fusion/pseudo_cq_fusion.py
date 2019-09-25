from ...multiview.additions import diversity_utils
from ..difficulty_fusion.difficulty_fusion import difficulty
from ..double_fault_fusion.double_fault_fusion import doubleFault


def genName(config):
    return "pseudo_cq_fusion"


def getBenchmark(benchmark, args=None):
    benchmark["multiview"]["pseudo_cq_fusion"] = ["take_everything"]
    return benchmark


def pseudoCQ(difficulty, doubleFlaut):
    return difficulty/float(doubleFlaut)


def getArgs(args, benchmark, views, viewsIndices, randomState, directory, resultsMonoview, classificationIndices):
    return diversity_utils.getArgs(args, benchmark, views,
                                   viewsIndices, randomState, directory,
                                   resultsMonoview, classificationIndices,
                                   [doubleFault, difficulty], "pseudo_cq_fusion")


def genParamsSets(classificationKWARGS, randomState, nIter=1):
    return diversity_utils.genParamsSets(classificationKWARGS, randomState, nIter=nIter)



class PseudoCQFusionClass(diversity_utils.DiversityFusionClass):

    def __init__(self, randomState, NB_CORES=1, **kwargs):
        diversity_utils.DiversityFusionClass.__init__(self, randomState, NB_CORES=1, **kwargs)

    def getSpecificAnalysis(self, classificationKWARGS):
        stringAnalysis = "Classifiers used for each view : "+ ', '.join(self.classifiersNames)+\
                         ', with a pseudo CQ of '+str(self.div_measure)
        return stringAnalysis