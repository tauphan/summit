from .additions.SelOptCB import SelfOptCBBoostClassifier

classifier_class_name = "SelfOptCBBoostBaseStump"

class SelfOptCBBoostBaseStump(SelfOptCBBoostClassifier):
    def __init__(self, n_max_iterations=10, random_state=42, twice_the_same=True,
                 random_start=False, save_train_data=True,
                 test_graph=True, base_estimator="LinearStump"):
        SelfOptCBBoostClassifier.__init__(self, n_max_iterations=n_max_iterations, random_state=random_state, twice_the_same=twice_the_same,
                 random_start=random_start, save_train_data=save_train_data,
                 test_graph=test_graph, base_estimator=base_estimator)


