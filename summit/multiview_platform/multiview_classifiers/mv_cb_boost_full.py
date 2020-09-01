from .additions.mv_cb_boost_adapt import MultiviewCBoundBoostingAdapt

classifier_class_name = "MVCBBoostFull"

class MVCBBoostFull(MultiviewCBoundBoostingAdapt):
    def __init__(self, n_max_iterations=10, random_state=None,
                 self_complemented=True, twice_the_same=False,
                 random_start=False, n_stumps=1, c_bound_sol=True,
                 estimators_generator="Stumps", mincq_tracking=False,
                 weight_add=3, weight_strategy="c_bound_based_dec",
                 weight_update="multiplicative", full_combination=True, **kwargs):
        MultiviewCBoundBoostingAdapt.__init__(self, n_max_iterations=n_max_iterations, random_state=random_state,
                 self_complemented=self_complemented, twice_the_same=twice_the_same,
                 random_start=random_start, n_stumps=n_stumps, c_bound_sol=c_bound_sol,
                 estimators_generator=estimators_generator, mincq_tracking=mincq_tracking,
                 weight_add=weight_add, weight_strategy=weight_strategy,
                 weight_update=weight_update, full_combination=full_combination, **kwargs)
