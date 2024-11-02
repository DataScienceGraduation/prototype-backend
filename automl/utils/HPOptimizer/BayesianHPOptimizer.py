from .BaseHPOptimizer import BaseHPOptimizer


class BayesianHPOptimizer(BaseHPOptimizer):
    def __init__(self, task, time_budget, metric, fast_mode, verbose):
        super().__init__(task, time_budget, metric, fast_mode, verbose)

    def fit(self, df, target_variable, fast_mode=False):
        pass