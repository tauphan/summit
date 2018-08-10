from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pickle

from .. import Metrics
from ..utils import HyperParameterSearch

# Author-Info
__author__ = "Nikolas Huelsmann, Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype
# __date__ = 2016 - 03 - 25


def randomizedSearch(X_train, y_train, randomState, outputFileName, classifierModule, CL_type, KFolds = 4, nbCores = 1,
    metric = ["accuracy_score", None], nIter = 30):

    estimator = getattr(classifierModule, CL_type)(randomState)
    params_dict = estimator.genDistribs()
    if params_dict:
        nb_possible_combinations = compute_possible_combinations(params_dict)
        metricModule = getattr(Metrics, metric[0])
        if metric[1] is not None:
            metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
        else:
            metricKWARGS = {}
        scorer = metricModule.get_scorer(**metricKWARGS)
        if nIter > nb_possible_combinations:
            nIter = nb_possible_combinations
        randomSearch = RandomizedSearchCV(estimator, n_iter=nIter, param_distributions=params_dict, refit=True,
                                          n_jobs=nbCores, scoring=scorer, cv=KFolds, random_state=randomState)
        detector = randomSearch.fit(X_train, y_train)

        bestParams = estimator.genBestParams(detector)

        scoresArray = detector.cv_results_['mean_test_score']
        params = estimator.genParamsFromDetector(detector)

        HyperParameterSearch.genHeatMaps(params, scoresArray, outputFileName)
        best_estimator = detector.best_estimator_
    else:
        best_estimator = estimator
        bestParams = {}
    testFoldsPreds = genTestFoldsPreds(X_train, y_train, KFolds, best_estimator)
    return bestParams, testFoldsPreds


def compute_possible_combinations(params_dict):
    n_possibs = np.ones(len(params_dict))*np.inf
    for value_index, value in enumerate(params_dict.values()):
        if type(value) == list:
            n_possibs[value_index] = len(value)
        elif isinstance(value, CustomRandint):
            n_possibs[value_index] = value.get_nb_possibilities()
    return np.prod(n_possibs)


def genTestFoldsPreds(X_train, y_train, KFolds, estimator):
    testFoldsPreds = []
    trainIndex = np.arange(len(y_train))
    folds = KFolds.split(trainIndex, y_train)
    foldLengths = np.zeros(KFolds.n_splits,dtype=int)
    for foldIndex, (trainIndices, testIndices) in enumerate(folds):
        foldLengths[foldIndex] = len(testIndices)
        estimator.fit(X_train[trainIndices], y_train[trainIndices])
        testFoldsPreds.append(estimator.predict(X_train[trainIndices]))
    minFoldLength = foldLengths.min()
    testFoldsPreds = np.array([testFoldPreds[:minFoldLength] for testFoldPreds in testFoldsPreds])
    return testFoldsPreds


class CustomRandint:
    """Used as a distribution returning a integer between low and high-1.
    It can be used with a multiplier agrument to be able to perform more complex generation
    for example 10 e -(randint)"""
    def __init__(self, low=0, high=0, multiplier=""):
        self.randint = randint(low, high)
        self.multiplier = multiplier

    def rvs(self, random_state=None):
        randinteger = self.randint.rvs(random_state=random_state)
        if self.multiplier == "e-":
            return 10 ** -randinteger
        else:
            return randinteger

    def get_nb_possibilities(self):
        return self.randint.b-self.randint.a


class CustomUniform:
    """Used as a distribution returning a float between loc and loc + scale..
        It can be used with a multiplier agrument to be able to perform more complex generation
        for example 10 e -(float)"""
    def __init__(self, loc=0, state=1, multiplier=""):
        self.uniform = uniform(loc, state)
        self.multiplier = multiplier

    def rvs(self, random_state=None):
        unif = self.uniform.rvs(random_state=random_state)
        if self.multiplier == 'e-':
            return 10 ** -unif
        else:
            return unif


class BaseMonoviewClassifier(object):

    def genBestParams(self, detector):
        return dict((param_name, detector.best_params_[param_name]) for param_name in self.param_names)

    def genParamsFromDetector(self, detector):
        if self.classed_params:
            classed_dict = dict((classed_param, get_names(detector.cv_results_["param_"+classed_param]))
                                for classed_param in self.classed_params)
        if self.param_names:
            return [(param_name, np.array(detector.cv_results_["param_"+param_name]))
                        if param_name not in self.classed_params else (param_name, classed_dict[param_name])
                        for param_name in self.param_names]
        else:
            return [()]

    def genDistribs(self):
        return dict((param_name, distrib) for param_name, distrib in zip(self.param_names, self.distribs))

    def getConfig(self):
        if self.param_names:
            return "\n\t\t- "+self.__class__.__name__+ "with "+ ", ".join([ param_name+" : " + self.to_str(param_name) for param_name in self.param_names])
        else:
            return "\n\t\t- "+self.__class__.__name__+ "with no config."

    def to_str(self, param_name):
        if param_name in self.weird_strings:
            if self.weird_strings[param_name] == "class_name":
                return self.get_params()[param_name].__class__.__name__
            else:
                return self.weird_strings[param_name](self.get_params()[param_name])
        else:
            return str(self.get_params()[param_name])

    def getFeatureImportance(self, directory, nb_considered_feats=50):
        """Used to generate a graph and a pickle dictionary representing feature importances"""
        featureImportances = self.feature_importances_
        sortedArgs = np.argsort(-featureImportances)
        featureImportancesSorted = featureImportances[sortedArgs][:nb_considered_feats]
        featureIndicesSorted = sortedArgs[:nb_considered_feats]
        fig, ax = plt.subplots()
        x = np.arange(len(featureIndicesSorted))
        formatter = FuncFormatter(percent)
        ax.yaxis.set_major_formatter(formatter)
        plt.bar(x, featureImportancesSorted)
        plt.title("Importance depending on feature")
        fig.savefig(directory + "feature_importances.png")
        plt.close()
        featuresImportancesDict = dict((featureIndex, featureImportance)
                                       for featureIndex, featureImportance in enumerate(featureImportances)
                                       if featureImportance != 0)
        with open(directory + 'feature_importances.pickle', 'wb') as handle:
            pickle.dump(featuresImportancesDict, handle)
        interpretString = "Feature importances : \n"
        for featureIndex, featureImportance in zip(featureIndicesSorted, featureImportancesSorted):
            if featureImportance > 0:
                interpretString += "- Feature index : " + str(featureIndex) + \
                                   ", feature importance : " + str(featureImportance) + "\n"
        return interpretString


def get_names(classed_list):
    return np.array([object_.__class__.__name__ for object_ in classed_list])

def percent(x, pos):
    """Used to print percentage of importance on the y axis"""
    return '%1.1f %%' % (x * 100)


class MonoviewResult(object):
    def __init__(self, view_index, classifier_name, view_name, metrics_scores, full_labels_pred,
                 classifier_config, y_test_multiclass_pred, test_folds_preds):
        self.view_index = view_index
        self.classifier_name = classifier_name
        self.view_name = view_name
        self.metrics_scores = metrics_scores
        self.full_labels_pred = full_labels_pred
        self.classifier_config = classifier_config
        self.y_test_multiclass_pred = y_test_multiclass_pred
        self.test_folds_preds = test_folds_preds




# def isUseful(labelSupports, index, CLASS_LABELS, labelDict):
#     if labelSupports[labelDict[CLASS_LABELS[index]]] != 0:
#         labelSupports[labelDict[CLASS_LABELS[index]]] -= 1
#         return True, labelSupports
#     else:
#         return False, labelSupports
#
#
# def getLabelSupports(CLASS_LABELS):
#     labels = set(CLASS_LABELS)
#     supports = [CLASS_LABELS.tolist().count(label) for label in labels]
#     return supports, dict((label, index) for label, index in zip(labels, range(len(labels))))
#
#
# def splitDataset(LABELS, NB_CLASS, LEARNING_RATE, DATASET_LENGTH, randomState):
#     validationIndices = extractRandomTrainingSet(LABELS, 1 - LEARNING_RATE, DATASET_LENGTH, NB_CLASS, randomState)
#     validationIndices.sort()
#     return validationIndices
#
#
# def extractRandomTrainingSet(CLASS_LABELS, LEARNING_RATE, DATASET_LENGTH, NB_CLASS, randomState):
#     labelSupports, labelDict = getLabelSupports(np.array(CLASS_LABELS))
#     nbTrainingExamples = [int(support * LEARNING_RATE) for support in labelSupports]
#     trainingExamplesIndices = []
#     usedIndices = []
#     while nbTrainingExamples != [0 for i in range(NB_CLASS)]:
#         isUseFull = False
#         index = int(randomState.randint(0, DATASET_LENGTH - 1))
#         if index not in usedIndices:
#             isUseFull, nbTrainingExamples = isUseful(nbTrainingExamples, index, CLASS_LABELS, labelDict)
#         if isUseFull:
#             trainingExamplesIndices.append(index)
#             usedIndices.append(index)
#     return trainingExamplesIndices


##### Generating Test and Train Data
# def calcTrainTestOwn(X,y,split):
#
#     classLabels = pd.Series(y)
#
#
#     data_train = []
#     data_test = []
#     label_train = []
#     label_test = []
#
#     # Reminder to store position in array
#     reminder = 0
#
#     for i in classLabels.unique():
#         # Calculate the number of samples per class
#         count = (len(classLabels[classLabels==i]))
#
#         # Min/Max: To determine the range to read from array
#         min_train = reminder
#         max_train = int(round(count * split)) +1 +reminder
#         min_test = max_train
#         max_test = count + reminder
#
#         #Extend the respective list with ClassLabels(y)/Features(X)
#         label_train.extend(classLabels[min_train:max_train])
#         label_test.extend(classLabels[min_test:max_test])
#         data_train.extend(X[min_train:max_train])
#         data_test.extend(X[min_test:max_test])
#
#         reminder = reminder + count
#
#     return np.array(data_train), np.array(data_test), np.array(label_train).astype(int), np.array(label_test).astype(int)

# def calcTrainTest(X,y,split):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split)
#
#     return (X_train, X_test, y_train, y_test)

# Classifiers

# ### Random Forest
#
# What are they?
# - Machine learning algorithm built for prediction tasks
#
# #### Pros:
# - Automatically model non-linear relations and interactions between variables. Perfect collinearity doesn't matter.
# - Easy to tune
# - Relatively easy to understand everything about them
# - Flexible enough to handle regression and classification tasks
# - Is useful as a step in exploratory data analysis
# - Can handle high dimensional data
# - Have a built in method of checking to see model accuracy
# - In general, beats most models at most prediction tasks
#
# #### Cons:
# - ?
#
#
# #### RF Algo
#
# The big idea: Combine a bunch of terrible decision trees into one awesome model.
#
# For each tree in the forest:
# 1. Take a bootstrap sample of the data
# 2. Randomly select some variables.
# 3. For each variable selected, find the split point which minimizes MSE (or Gini Impurity or Information Gain if classification).
# 4. Split the data using the variable with the lowest MSE (or other stat).
# 5. Repeat step 2 through 4 (randomly selecting new sets of variables at each split) until some stopping condition is satisfied or all the data is exhausted.
#
# Repeat this process to build several trees.
#
# To make a prediction, run an observation down several trees and average the predicted values from all the trees (for regression) or find the most popular class predicted (if classification)
#
# #### Most important parameters (and what they mean)
#
# - **Parameters that make the model better**
#     - **n_estimators:** Number of Trees. Choose a number as high as your computer can handle
#     - **max_features:** Number of features to consider for the best split: Here all!
#     - **min_samples_leaf:** Minimum number of samples in newly created leaves: Try [1,2,3]. If 3 is best: try higher numbers
# - **Parameters that will make it easier to train your model**
#     - **n_jobs:** Number of used CPU's. -1==all. Use %timeit to see speed improvement
#         - **Problem:** Nikolas PC -> error with multiple CPU...
#     - **random_state:** Set to 42 if you want others to replicate your results
#     - **oob_score:** Random Forest Validation method: out-of-bag predictions
#
# #### OOB Predictions
# About a third of observations don't show up in a bootstrap sample.
#
# Because an individual tree in the forest is made from a bootstrap sample, it means that about a third of the data was not used to build that tree. We can track which observations were used to build which trees.
#
# **Here is the magic.**
#
# After the forest is built, we take each observation in the dataset and identify which trees used the observation and which trees did not use the observation (based on the bootstrap sample). We use the trees the observation was not used to build to predict the true value of the observation. About a third of the trees in the forest will not use any specific observation from the dataset.
#
# OOB predictions are similar to following awesome, but computationally expensive method:
#
# 1. Train a model with n_estimators trees, but exclude one observation from the dataset.
# 2. Use the trained model to predict the excluded observation. Record the prediction.
# 3. Repeat this process for every single observation in the dataset.
# 4. Collect all your final predictions. These will be similar to your oob prediction errors.
#
# The leave-one-out method will take n_estimators*time_to_train_one_model*n_observations to run.
#
# The oob method will take n_estimators x(times) time_to_train_one_model x(times) 3 to run (the x(times)3 is because if you want to get an accuracy estimate of a 100 tree forest, you will need to train 300 trees. Why? Because with 300 trees each observation will have about 100 trees it was not used to build that can be used for the oob_predictions).
#
# This means the oob method is n_observations/3 times faster to train then the leave-one-out method.
#

# X_test: Test Data
# y_test: Test Labels
# num_estimators: number of trees
# def MonoviewClassifRandomForest(X_train, y_train, nbFolds=4, nbCores=1, **kwargs):
#     num_estimators = kwargs["classifier__n_estimators"]
#     # PipeLine with RandomForest classifier
#     pipeline_rf = Pipeline([('classifier', RandomForestClassifier())])
#
#     # Parameters for GridSearch: Number of Trees
#     # can be extended with: oob_score, min_samples_leaf, max_features
#     param_rf = kwargs
#
#     # pipeline: Gridsearch avec le pipeline comme estimator
#     # param: pour obtenir le meilleur model il va essayer tous les possiblites
#     # refit: pour utiliser le meilleur model apres girdsearch
#     # n_jobs: Nombre de CPU (Mon ordi a des problemes avec -1 (Bug Python 2.7 sur Windows))
#     # scoring: scoring...
#     # cv: Nombre de K-Folds pour CV
#     grid_rf = GridSearchCV(
#         pipeline_rf,
#         param_grid=param_rf,
#         refit=True,
#         n_jobs=nbCores,
#         scoring='accuracy',
#         cv=nbFolds,
#     )
#
#     rf_detector = grid_rf.fit(X_train, y_train)
#
#     desc_estimators = [rf_detector.best_params_["classifier__n_estimators"]]
#     description = "Classif_" + "RF" + "-" + "CV_" + str(nbFolds) + "-" + "Trees_" + str(map(str, desc_estimators))
#
#     return description, rf_detector
#
#
# def MonoviewClassifSVMLinear(X_train, y_train, nbFolds=4, nbCores=1, **kwargs):
#     pipeline_SVMLinear = Pipeline([('classifier', sklearn.svm.SVC())])
#     param_SVMLinear = kwargs
#
#     grid_SVMLinear = GridSearchCV(pipeline_SVMLinear, param_grid=param_SVMLinear, refit=True, n_jobs=nbCores,
#                                   scoring='accuracy',
#                                   cv=nbFolds)
#     SVMLinear_detector = grid_SVMLinear.fit(X_train, y_train)
#     desc_params = [SVMLinear_detector.best_params_["classifier__C"]]
#     description = "Classif_" + "SVC" + "-" + "CV_" + str(nbFolds) + "-" + "-".join(map(str, desc_params))
#     return description, SVMLinear_detector
#
#
# def MonoviewClassifSVMRBF(X_train, y_train, nbFolds=4, nbCores=1, **kwargs):
#     pipeline_SVMRBF = Pipeline([('classifier', sklearn.svm.SVC())])
#     param_SVMRBF = kwargs
#
#     grid_SVMRBF = GridSearchCV(pipeline_SVMRBF, param_grid=param_SVMRBF, refit=True, n_jobs=nbCores, scoring='accuracy',
#                                cv=nbFolds)
#     SVMRBF_detector = grid_SVMRBF.fit(X_train, y_train)
#     desc_params = [SVMRBF_detector.best_params_["classifier__C"]]
#     description = "Classif_" + "SVC" + "-" + "CV_" + str(nbFolds) + "-" + "-".join(map(str, desc_params))
#     return description, SVMRBF_detector
#
#
# def MonoviewClassifDecisionTree(X_train, y_train, nbFolds=4, nbCores=1, **kwargs):
#     pipeline_DT = Pipeline([('classifier', sklearn.tree.DecisionTreeClassifier())])
#     param_DT = kwargs
#
#     grid_DT = GridSearchCV(pipeline_DT, param_grid=param_DT, refit=True, n_jobs=nbCores, scoring='accuracy',
#                            cv=nbFolds)
#     DT_detector = grid_DT.fit(X_train, y_train)
#     desc_params = [DT_detector.best_params_["classifier__max_depth"]]
#     description = "Classif_" + "DT" + "-" + "CV_" + str(nbFolds) + "-" + "-".join(map(str, desc_params))
#     return description, DT_detector
#
#
# def MonoviewClassifSGD(X_train, y_train, nbFolds=4, nbCores=1, **kwargs):
#     pipeline_SGD = Pipeline([('classifier', sklearn.linear_model.SGDClassifier())])
#     param_SGD = kwargs
#     grid_SGD = GridSearchCV(pipeline_SGD, param_grid=param_SGD, refit=True, n_jobs=nbCores, scoring='accuracy',
#                             cv=nbFolds)
#     SGD_detector = grid_SGD.fit(X_train, y_train)
#     desc_params = [SGD_detector.best_params_["classifier__loss"], SGD_detector.best_params_["classifier__penalty"],
#                    SGD_detector.best_params_["classifier__alpha"]]
#     description = "Classif_" + "Lasso" + "-" + "CV_" + str(nbFolds) + "-" + "-".join(map(str, desc_params))
#     return description, SGD_detector
#
#
# def MonoviewClassifKNN(X_train, y_train, nbFolds=4, nbCores=1, **kwargs):
#     pipeline_KNN = Pipeline([('classifier', sklearn.neighbors.KNeighborsClassifier())])
#     param_KNN = kwargs
#     grid_KNN = GridSearchCV(pipeline_KNN, param_grid=param_KNN, refit=True, n_jobs=nbCores, scoring='accuracy',
#                             cv=nbFolds)
#     KNN_detector = grid_KNN.fit(X_train, y_train)
#     desc_params = [KNN_detector.best_params_["classifier__n_neighbors"]]
#     description = "Classif_" + "Lasso" + "-" + "CV_" + str(nbFolds) + "-" + "-".join(map(str, desc_params))
#     return description, KNN_detector



    # def calcClassifRandomForest(X_train, X_test, y_test, y_train, num_estimators):
    #    from sklearn.grid_search import ParameterGrid
    #    param_rf = { 'classifier__n_estimators': num_estimators}
    #    forest = RandomForestClassifier()
    #
    #    bestgrid=0;
    #    for g in ParameterGrid(grid):
    #        forest.set_params(**g)
    #        forest.fit(X_train,y_train)
    #        score = forest.score(X_test, y_test)
    #
    #        if score > best_score:
    #            best_score = score
    #            best_grid = g
    #
    #    rf_detector = RandomForestClassifier()
    #    rf_detector.set_params(**best_grid)
    #    rf_detector.fit(X_train,y_train)

    #    #desc_estimators = best_grid
    #    description = "Classif_" + "RF" + "-" + "CV_" +  "NO" + "-" + "Trees_" + str(best_grid)

    #    return (description, rf_detector)
