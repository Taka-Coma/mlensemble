from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier as kNN
from metric_learn import LMNN
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.pipeline import Pipeline


class MLBaggingClassifier(BalancedBaggingClassifier):
    def __init__(self,
        sampling_strategy='auto',
        metric_learner=LMNN(),
        base_estimator=kNN(),
        n_jobs=1,
        n_estimators=10,
		n_features=None
        ):
        super().__init__(
            n_estimators=n_estimators,
            max_samples=1.0,
            max_features=1.0,
			replacement=True,
            bootstrap=True,
            bootstrap_features=False,
            oob_score=False,
            n_jobs=n_jobs,
            sampling_strategy=sampling_strategy,
        )
        self.n_features_in_ = n_features
        self.base_estimator = Pipeline([
            ('metric learner', clone(metric_learner)),
            ('classifier', clone(base_estimator))
        ])



