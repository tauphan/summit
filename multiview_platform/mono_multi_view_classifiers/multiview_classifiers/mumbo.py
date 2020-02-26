from sklearn.tree import DecisionTreeClassifier
import numpy as np

from multimodal.boosting.mumbo import MumboClassifier
from ..multiview.multiview_utils import BaseMultiviewClassifier
from ..utils.hyper_parameter_search import CustomRandint
from ..utils.dataset import get_examples_views_indices

classifier_class_name = "Mumbo"

class Mumbo(BaseMultiviewClassifier, MumboClassifier):

    def __init__(self, base_estimator=None,
                 n_estimators=50,
                 random_state=None,
                 best_view_mode="edge"):
        BaseMultiviewClassifier.__init__(self, random_state)
        MumboClassifier.__init__(self, base_estimator=base_estimator,
                                    n_estimators=n_estimators,
                                    random_state=random_state,
                                    best_view_mode=best_view_mode)
        self.param_names = ["base_estimator", "n_estimators", "random_state", "best_view_mode"]
        self.distribs = [[DecisionTreeClassifier(max_depth=1),
                          DecisionTreeClassifier(max_depth=2),
                          DecisionTreeClassifier(max_depth=3),
                          DecisionTreeClassifier(max_depth=4)],
                         CustomRandint(5,200), [random_state], ["edge", "error"]]

    def fit(self, X, y, train_indices=None, view_indices=None):
        train_indices, view_indices = get_examples_views_indices(X,
                                                                 train_indices,
                                                                 view_indices)
        numpy_X, view_limits = X.to_numpy_array(example_indices=train_indices,
                                                view_indices=view_indices)
        return MumboClassifier.fit(self, numpy_X, y[train_indices],
                                                view_limits)

    def predict(self, X, example_indices=None, view_indices=None):
        example_indices, view_indices = get_examples_views_indices(X,
                                                                 example_indices,
                                                                 view_indices)
        numpy_X, view_limits = X.to_numpy_array(example_indices=example_indices,
                                                view_indices=view_indices)
        return MumboClassifier.predict(self, numpy_X)

    def get_interpretation(self, directory, labels, multiclass=False):
        intepret_string = "Mumbo used "+str(len(self.best_views_)) +" iterations to converge, selecting views : \n" + ", ".join(map(str, self.best_views_)) + "\n\n With estimator weights : \n"+ "\n".join(map(str,self.estimator_weights_/np.sum(self.estimator_weights_)))
        return intepret_string
