import h5py
import numpy as np
import sys
import Multiview
import Metrics

def searchBestSettings(dataset, classifierName, metrics, iLearningIndices, iKFolds, viewsIndices=None, searchingTool="gridSearch", nIter=1, **kwargs):
    if viewsIndices is None:
        viewsIndices = range(dataset.get("Metadata").attrs["nbView"])
    thismodule = sys.modules[__name__]
    searchingToolMethod = getattr(thismodule, searchingTool)
    bestSettings = searchingToolMethod(dataset, classifierName, metrics, iLearningIndices, iKFolds, viewsIndices=viewsIndices, nIter=nIter, **kwargs)
    return bestSettings # or well set clasifier ?


def gridSearch(dataset, classifierName, viewsIndices=None, kFolds=None, nIter=1, **kwargs):
    #si grid search est selectionne, on veut tester certaines valeurs
    pass


def randomizedSearch(dataset, classifierName, metrics, iLearningIndices, iKFolds, viewsIndices=None, nIter=1, nbCores=1, **classificationKWARGS):
    if viewsIndices is None:
        viewsIndices = range(dataset.get("Metadata").attrs["nbView"])
    metric = metrics[0]
    metricModule = getattr(Metrics, metric[0])
    if metric[1]!=None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    else:
        metricKWARGS = {}
    classifierPackage =getattr(Multiview,classifierName)  # Permet d'appeler un module avec une string
    classifierModule = getattr(classifierPackage, classifierName)
    classifierClass = getattr(classifierModule, classifierName)
    statsIter = len(iLearningIndices)
    if classifierName != "Mumbo":
        datasetLength = dataset.get("Metadata").attrs["datasetLength"]
        paramsSets = classifierModule.genParamsSets(classificationKWARGS, nIter=nIter)
        for paramsSet in paramsSets:
            if metricModule.getConfig()[-14]=="h":
                baseScore = -1000.0
                isBetter = "higher"
            else:
                baseScore = 1000.0
                isBetter = "lower"
            scores = []
            for statsIterIndex in range(statsIter):
                for fold in iKFolds[statsIterIndex]:
                    fold.sort()
                    trainIndices = [index for index in range(datasetLength) if (index not in fold) and (index in iLearningIndices[statsIterIndex])]
                    classifier = classifierClass(NB_CORES=nbCores, **classificationKWARGS)
                    classifier.setParams(paramsSet)
                    classifier.fit_hdf5(dataset, trainIndices=trainIndices, viewsIndices=viewsIndices)
                    trainLabels = classifier.predict_hdf5(dataset, usedIndices=trainIndices, viewsIndices=viewsIndices)
                    testLabels = classifier.predict_hdf5(dataset, usedIndices=fold, viewsIndices=viewsIndices)
                    trainScore = metricModule.score(dataset.get("Labels").value[trainIndices], trainLabels)
                    testScore = metricModule.score(dataset.get("Labels").value[fold], testLabels)
                    scores.append(testScore)
            crossValScore = np.mean(np.array(scores))

            if isBetter=="higher" and crossValScore>baseScore:
                baseScore = crossValScore
                bestSettings = paramsSet
            elif isBetter=="lower" and crossValScore<baseScore:
                baseScore = crossValScore
                bestSettings = paramsSet
        classifier = classifierClass(NB_CORES=nbCores, **classificationKWARGS)
        classifier.setParams(bestSettings)

    else:
        bestConfigs, _ = classifierModule.gridSearch_hdf5(dataset, viewsIndices, classificationKWARGS, iLearningIndices[0], metric=metric, nIter=nIter)
        classificationKWARGS["classifiersConfigs"] = bestConfigs
        classifier = classifierClass(NB_CORES=nbCores, **classificationKWARGS)

    return classifier


def spearMint(dataset, classifierName, viewsIndices=None, kFolds=None, nIter=1, **kwargs):
    pass

# nohup python ~/dev/git/spearmint/spearmint/main.py . &

# import json
# import numpy as np
# import math
#
# from os import system
# from os.path import join
#
#
# def run_kover(dataset, split, model_type, p, max_rules, output_dir):
#     outdir = join(output_dir, "%s_%f" % (model_type, p))
#     kover_command = "kover learn " \
#                     "--dataset '%s' " \
#                     "--split %s " \
#                     "--model-type %s " \
#                     "--p %f " \
#                     "--max-rules %d " \
#                     "--max-equiv-rules 10000 " \
#                     "--hp-choice cv " \
#                     "--random-seed 0 " \
#                     "--output-dir '%s' " \
#                     "--n-cpu 1 " \
#                     "-v" % (dataset,
#                             split,
#                             model_type,
#                             p,
#                             max_rules,
#                             outdir)
#
#     system(kover_command)
#
#     return json.load(open(join(outdir, "results.json")))["cv"]["best_hp"]["score"]
#
#
# def main(job_id, params):
#     print params
#
#     max_rules = params["MAX_RULES"][0]
#
#     species = params["SPECIES"][0]
#     antibiotic = params["ANTIBIOTIC"][0]
#     split = params["SPLIT"][0]
#
#     model_type = params["model_type"][0]
#
#     # LS31
#     if species == "saureus":
#         dataset_path = "/home/droale01/droale01-ls31/projects/genome_scm/data/earle_2016/saureus/kover_datasets/%s.kover" % antibiotic
#     else:
#         dataset_path = "/home/droale01/droale01-ls31/projects/genome_scm/genome_scm_paper/data/%s/%s.kover" % (species, antibiotic)
#
#     output_path = "/home/droale01/droale01-ls31/projects/genome_scm/manifold_scm/spearmint/vanilla_scm/%s/%s" % (species, antibiotic)
#
#     # MacBook
#     #dataset_path = "/Volumes/Einstein 1/kover_phylo/datasets/%s/%s.kover" % (species, antibiotic)
#     #output_path = "/Volumes/Einstein 1/manifold_scm/version2/%s_spearmint" % antibiotic
#
#     return run_kover(dataset=dataset_path,
#                      split=split,
#                      model_type=model_type,
#                      p=params["p"][0],
#                      max_rules=max_rules,
#                      output_dir=output_path)
# killall mongod && sleep 1 && rm -r database/* && rm mongo.log*
# mongod --fork --logpath mongo.log --dbpath database
#
# {
#     "language"        : "PYTHON",
#     "experiment-name" : "vanilla_scm_cdiff_azithromycin",
#     "polling-time"    : 1,
#     "resources" : {
#         "my-machine" : {
#             "scheduler"         : "local",
#             "max-concurrent"    : 5,
#             "max-finished-jobs" : 100
#         }
#     },
#     "tasks": {
#         "resistance" : {
#             "type"       : "OBJECTIVE",
#             "likelihood" : "NOISELESS",
#             "main-file"  : "spearmint_wrapper",
#             "resources"  : ["my-machine"]
#         }
#     },
#     "variables": {
#
#         "MAX_RULES" : {
#             "type" : "ENUM",
#             "size" : 1,
#             "options": [10]
#         },
#
#
#         "SPECIES" : {
#             "type" : "ENUM",
#             "size" : 1,
#             "options": ["cdiff"]
#         },
#         "ANTIBIOTIC" : {
#             "type" : "ENUM",
#             "size" : 1,
#             "options": ["azithromycin"]
#         },
#         "SPLIT" : {
#             "type" : "ENUM",
#             "size" : 1,
#             "options": ["split_seed_2"]
#         },
#
#
#         "model_type" : {
#             "type" : "ENUM",
#             "size" : 1,
#             "options": ["conjunction", "disjunction"]
#         },
#         "p" : {
#             "type" : "FLOAT",
#             "size" : 1,
#             "min"  : 0.01,
#             "max"  : 100
#         }
#     }
# }