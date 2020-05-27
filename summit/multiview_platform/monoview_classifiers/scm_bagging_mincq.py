from .scm_bagging import ScmBagging
from ..utils.hyper_parameter_search import CustomUniform, CustomRandint

classifier_class_name = "ScmBaggingMinCq"

class ScmBaggingMinCq(ScmBagging):
    def __init__(self,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 max_rules=10,
                 p_options=[0.316],
                 model_type="conjunction",
                 min_cq_combination=True,
                 min_cq_mu = 10e-3,
                 random_state=None):
        ScmBagging.__init__(self, n_estimators=n_estimators,
                 max_samples=max_samples,
                 max_features=max_features,
                 max_rules=max_rules,
                 p_options=p_options,
                 model_type=model_type,
                 min_cq_combination=min_cq_combination,
                 min_cq_mu=min_cq_mu,
                 random_state=random_state)
        self.param_names.append("min_cq_mu")
        self.distribs.append(CustomRandint(1,7, multiplier='e-'))