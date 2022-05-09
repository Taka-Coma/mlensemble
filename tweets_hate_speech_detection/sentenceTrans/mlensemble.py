import numbers

import numpy as np

from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from metric_learn import LMNN

from imblearn.under_sampling import RandomUnderSampler
from imblearn.utils import check_target_type, check_sampling_strategy
from imblearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler


class MLEnsembleClassifier(BaggingClassifier):
    def __init__(
        self,
        n_estimators=10,
        metric_learner=LMNN(),
        base_estimator=None,
        *,
        warm_start=False,
        sampling_strategy="auto",
        replacement=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=False,
            bootstrap_features=False,
            oob_score=False,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        self.sampling_strategy = sampling_strategy
        self.replacement = replacement
        self.metric_learner = metric_learner 


    def _validate_y(self, y):
        y_encoded = super()._validate_y(y)
        if isinstance(self.sampling_strategy, dict):
            self._sampling_strategy = {
                np.where(self.classes_ == key)[0][0]: value
                for key, value in check_sampling_strategy(
                    self.sampling_strategy, y, 'under-sampling',
                ).items()
            }
        else:
            self._sampling_strategy = self.sampling_strategy
        return y_encoded

    def _validate_estimator(self, default=AdaBoostClassifier()):
        """Check the estimator and the n_estimator attribute, set the
        `base_estimator_` attribute."""
        if not isinstance(self.n_estimators, (numbers.Integral, np.integer)):

            raise ValueError(
                "n_estimators must be an integer, "
                "got {}.".format(type(self.n_estimators))
            )

        if self.n_estimators <= 0:
            raise ValueError(
                "n_estimators must be greater than zero, "
                "got {}.".format(self.n_estimators)
            )

        if self.base_estimator is not None:
            base_estimator = clone(self.base_estimator)
        else:
            base_estimator = clone(default)

        self.sampler = RandomUnderSampler(
                        sampling_strategy=self._sampling_strategy,
                        replacement=self.replacement,
                    )

        self.base_estimator_ = Pipeline(
            [
                #('scaler', StandardScaler()),
                ("metric_learner", self.metric_learner),
                ("classifier", base_estimator),
            ]
        )

    def fit(self, X, y):
        """Train the ensemble on the training set.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        check_target_type(y)
        # RandomUnderSampler is not supporting sample_weight. We need to pass
        # None.
        return self._fit(X, y, self.max_samples, sample_weight=None)
