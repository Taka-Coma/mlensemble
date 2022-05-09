import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from sklearn.base import clone

from metric_learn import LMNN

from imblearn.ensemble import EasyEnsembleClassifier 
from sklearn.neighbors import KNeighborsClassifier as kNN
from mlensemble import MLEnsembleClassifier as MLE

from imblearn.under_sampling import RandomUnderSampler as RUS

from utils import PRF_TPR_TNR_gmean_AUC, prepareData


class MLBoost():
    def __init__(
        self,
        metric_learner=LMNN(),
        base_estimator=None,
        sampling_strategy=0.3,
        replacement=False,
        n_jobs=-1,
        random_state=None,
        verbose=0,
        max_iteration=10
    ):
        self.sampling_strategy = sampling_strategy
        self.replacement = replacement
        self.metric_learner = metric_learner 
        self.n_jobs = n_jobs
        self.max_iteration = max_iteration

        if base_estimator is None:
            self.base_estimator = kNN()
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
        scores = []

        failed_samples = False
        for _ in range(self.max_iteration):
            if not failed_samples:
                X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5)

            undersampler = RUS(sampling_strategy=self.sampling_strategy)
            X_train_us, y_train_us = undersampler.fit_resample(X_train, y_train)

            metric_learner = clone(self.metric_learner)
            metric_learner.fit(X_train_us, y_train_us)
            self.ml_array.append(metric_learner)

            X_train_, X_valid_ = None, None
            for ml in self.ml_array:
                if X_train_ is None:
                    X_train_ = ml.transform(X_train)
                    X_valid_ = ml.transform(X_valid)
                else:
                    X_train_ = np.hstack((X_train_, ml.transform(X_train)))
                    X_valid_ = np.hstack((X_valid_, ml.transform(X_valid)))

            #classifier = clone(self.base_estimator)
            classifier = EasyEnsembleClassifier(n_jobs=self.n_jobs)
            classifier.fit(X_train_, y_train)

            y_preds = classifier.predict(X_valid_) 
            y_preds_proba = classifier.predict_proba(X_valid_) 

            score = PRF_TPR_TNR_gmean_AUC(y_valid, y_preds, y_preds_proba)
            scores.append(score)
            #print(score)

            correct_mask = [s==t for s, t in zip(y_valid, y_preds)]
            failed_ids = [i for i, _ in enumerate(correct_mask) if _]
            success_ids = [i for i, _ in enumerate(correct_mask) if not _]

            X_valid_plus = X_valid[failed_ids,:]
            y_valid_plus = [y_valid[i] for i in failed_ids]

            # extract succeeded-to-classify samples
            X_valid_minus = X_valid[success_ids,:]
            y_valid_minus = [y_valid[i] for i in success_ids]

            X_sup = np.vstack((X_train, X_valid_minus))
            y_sup = np.hstack((y_train, y_valid_minus))

            num_samples = len(y_train) - len(y_valid_plus)
            sample_indices = np.random.choice(len(y_sup), num_samples, replace=False)

            if len(failed_ids) > 0:
            	X_train = np.vstack((X_valid_plus, X_sup[sample_indices, :]))
            	y_train = np.hstack((y_valid_plus, y_sup[sample_indices]))
            else:
            	X_train = X_sup[sample_indices, :]
            	y_train = y_sup[sample_indices]

            X_valid = np.delete(X_sup, sample_indices, 0)
            y_valid = np.delete(y_sup, sample_indices)

            failed_samples = True

        #print(scores)

        #metric_learner = clone(self.metric_learner)
        #X_ = self.transform_base(X)
        #X_last = metric_learner.fit_transform(X_, y)
        #self.last_metric_learner = metric_learner

        X_last = self.transform_base(X)

        #classifier = clone(self.base_estimator)
        classifier = EasyEnsembleClassifier(n_jobs=self.n_jobs)
        #classifier = MLE(n_jobs=self.n_jobs)
        classifier.fit(X_last, y)

        self.classifier = classifier
        return self


    def predict(self, X):
        X_ = self.transform(X)
        return self.classifier.predict(X_)

    def predict_proba(self, X):
        X_ = self.transform(X)
        return self.classifier.predict_proba(X_)

    def transform(self, X):
        X_last = self.transform_base(X)
        #X_ = self.transform_base(X)
        #X_last = self.last_metric_learner.transform(X_)
        return X_last

    def transform_base(self, X):
        X_ = None
        for ml in self.ml_array:
            if X_ is None:
                X_ = ml.transform(X)
            else:
                X_ = np.hstack((X_, ml.transform(X)))
        return X_
