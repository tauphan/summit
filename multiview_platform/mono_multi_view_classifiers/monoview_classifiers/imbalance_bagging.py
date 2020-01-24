from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from ..monoview.monoview_utils import BaseMonoviewClassifier, CustomRandint, CustomUniform


classifier_class_name = "ImbalanceBagging"

class ImbalanceBagging(BaseMonoviewClassifier, BalancedBaggingClassifier):

    def __init__(self, random_state=None, base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=10,
                 sampling_strategy="auto", replacement=False,):
        super(ImbalanceBagging, self).__init__(random_state=random_state, base_estimator=base_estimator,
                                         n_estimators=n_estimators,
                                         sampling_strategy=sampling_strategy,
                                         replacement=replacement)

        self.param_names = ["n_estimators", "base_estimator", "sampling_strategy",]
        self.classed_params = ["base_estimator"]
        self.distribs = [CustomRandint(low=1, high=50),
                         [DecisionTreeClassifier(max_depth=1)],
                         ["auto"]]
        self.weird_strings = {"base_estimator": "class_name"}



