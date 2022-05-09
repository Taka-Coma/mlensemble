import numbers

import numpy as np

from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier
from metric_learn import LMNN

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.ensemble import EasyEnsembleClassifier

from sklearn.preprocessing import StandardScaler


class MLStackingClassifier():
    def __init__(
        self,
        n_estimators=10,
        metric_learner=LMNN(),
        base_estimator=None,
        sampling_strategy="auto",
        replacement=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.sampling_strategy = sampling_strategy
        self.replacement = replacement
        self.metric_learner = metric_learner 
        self.n_jobs = n_jobs

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

        estimators = [
            (f'clf_{i}', Pipeline([
                ('sampler', RandomUnderSampler(sampling_strategy=self.sampling_strategy)),
                ('metric learner', LMNN()),
                ('classifier', AdaBoostClassifier())
            ]))
            for i in range(self.n_estimators)
        ]
        self.estimator = StackingClassifier(
            estimators,
            final_estimator=EasyEnsembleClassifier(),
            passthrough=True,
            n_jobs=self.n_jobs)
        self.estimator.fit(X, y)

        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        return self.estimator.predict(X)
