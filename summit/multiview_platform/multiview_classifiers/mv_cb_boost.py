from .additions.mv_cb_boost_adapt import MultiviewCBoundBoostingAdapt

classifier_class_name = "MVCBBoost"

class MVCBBoost(MultiviewCBoundBoostingAdapt):
    def __init__(self, n_estimators=100,
                              random_state=42,
                              self_complemented=True,
                              twice_the_same=False,
                              random_start=False,
                              n_stumps=10,
                              c_bound_sol=True,
                              base_estimator="Trees",
                              max_depth=1,
                              mincq_tracking=False,
                              weight_add=3,
                              weight_strategy="c_bound_based_dec",
                              weight_update="multiplicative",
                              full_combination=False,
                              min_cq_pred=False,
                              min_cq_mu=10e-3,
                              sig_mult=15,
                              sig_offset=5,
                              use_previous_voters=False, **kwargs):
        MultiviewCBoundBoostingAdapt.__init__(self, n_estimators=n_estimators, random_state=random_state,
                 self_complemented=self_complemented, twice_the_same=twice_the_same,
                 random_start=random_start, n_stumps=n_stumps, c_bound_sol=c_bound_sol, max_depth=max_depth,
                 base_estimator=base_estimator, mincq_tracking=mincq_tracking,
                 weight_add=weight_add, weight_strategy=weight_strategy,
                 weight_update=weight_update, use_previous_voters=use_previous_voters,
                                         full_combination=full_combination,
                                         min_cq_pred=min_cq_pred, min_cq_mu=min_cq_mu,
                                         sig_mult=sig_mult, sig_offset=sig_offset, **kwargs)
        # self.param_names+=["weight_update", "weight_strategy"]
        # self.distribs+=[["multiplicative", "additive", "replacement"],["c_bound_based_broken", "c_bound_based", "c_bound_based_dec", "sigmoid"]]