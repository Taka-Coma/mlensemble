import numpy as np

from sklearn.base import clone

from metric_learn import LMNN

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from imblearn.ensemble import RUSBoostClassifier


class MLBoostingClassifier():
    def __init__(
        self,
        metric_learner=LMNN(),
        base_estimator=None,
        random_state=None,
        verbose=0,
    ):
        self.metric_learner = metric_learner 

        if base_estimator is None:
            self.base_estimator = DecisionTreeClassifier(max_depth=1)
        else:
            self.base_estimator = base_estimator

        self.ml_array = []


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

        return self._fit(X, y)


    def _fit(self, X, y):
        classifier = RUSBoostClassifier(
			base_estimator=PipelineSW([
				('metric learner', clone(self.metric_learner)),
				('classifier', clone(self.base_estimator))
			]),
			n_estimators=50)
        classifier.fit(X, y)

        self.classifier = classifier
        return self


    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)


class PipelineSW(Pipeline):
	def fit(self, X, y, sample_weight=None):
		"""Fit and pass sample weights only to the last step"""
		if sample_weight is not None:
			kwargs = {self.steps[-1][0] + '__sample_weight': sample_weight}
		else:
			kwargs = {}
		return super().fit(X, y, **kwargs)
