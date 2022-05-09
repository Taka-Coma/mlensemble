# -*- coding: utf-8 -*-
import warnings

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition

from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import INPUT, UNSIGNED_DATA, DENSE
from autosklearn.util.common import check_for_bool, check_none


class LMNN(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, k=None, random_state=None):
        self.k = k
        self.random_state = random_state

    def fit(self, X, Y=None):
        from metric_learn import LMNN

        if check_none(self.k):
            self.k = None
        else:
            self.k = int(self.k)

        self.preprocessor = LMNN(
            k=self.k, random_state=self.random_state
        )
        # Make the RuntimeWarning an Exception!
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message='array must not contain infs or NaNs')
            try:
                self.preprocessor.fit(X)
            except ValueError as e:
                if 'array must not contain infs or NaNs' in e.args[0]:
                    raise ValueError("Bug in scikit-learn: "
                                     "https://github.com/scikit-learn/scikit-learn/pull/2738")

        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LMNN',
                'name': 'Large Margin Nearest Neighbor',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'handles_multioutput': True,
                'is_deterministic': False,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (INPUT, UNSIGNED_DATA)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        k = UniformIntegerHyperparameter("k", 2, 50, default_value=5)
        cs.add_hyperparameters([k])

        #cs.add_condition(EqualsCondition(k, whiten, "True"))

        return cs
