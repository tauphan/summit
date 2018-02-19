import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pickle


def percent(x, pos):
    """Used to print percentage of importance on the y axis"""
    return '%1.1f %%' % (x * 100)


def getFeatureImportance(classifier, directory, interpretString=""):
    """Used to generate a graph and a pickle dictionary representing feature importances"""
    featureImportances = classifier.feature_importances_
    sortedArgs = np.argsort(-featureImportances)
    featureImportancesSorted = featureImportances[sortedArgs][:50]
    featureIndicesSorted = sortedArgs[:50]
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
    with open(directory+'feature_importances.pickle', 'wb') as handle:
        pickle.dump(featuresImportancesDict, handle)
    interpretString += "Feature importances : \n"
    for featureIndex, featureImportance in zip(featureIndicesSorted, featureImportancesSorted):
        if featureImportance>0:
            interpretString+="- Feature index : "+str(featureIndex)+\
                             ", feature importance : "+str(featureImportance)+"\n"
    return interpretString