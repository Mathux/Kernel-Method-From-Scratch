from src.tools.utils import Logger
import numpy as np


class KMethod(Logger):
    def __init__(self, kernel, name="NO NAME", verbose=True):
        self.__name__ = name
        self.verbose = verbose
        self._alpha = None
        self.kernel = kernel

    # Load the dataset (if there are one) in the kernel
    # or just load the labels
    def load_dataset(self, dataset=None, labels=None):
        if dataset is None:
            if labels is None:
                self._log("Taking data from the kernel directly")
                self.labels = self.kernel.labels
            else:
                self.labels = labels
        else:
            self._log("Load the data in the kernel")
            self.kernel.dataset = dataset
            self.labels = dataset.labels
        
    def predict(self, x):
        K_xi = self.kernel.predict(x)
        return self.alpha.dot(K_xi)
        # return 1 if sigmoid(self.alpha.dot(K_xi)) >= 0.5 else -1
        # return 1 if self.alpha.dot(K_xi) >= 0.5 else 0
        
    def score_recall_precision(self, X, y):
        predictions = self.predict(X)
        tp = np.sum((predictions == 1.) * (y == 1.))
        fn = np.sum((predictions == -1.) * (y == 1.))
        fp = np.sum((predictions == 1.) * (y == -1.))
        recall = tp / (fn + tp)
        precision = tp / (fp + tp)
        return np.sum(y == predictions) / X.shape[0], recall, precision
            
    @property
    def alpha(self):
        if self._alpha is None:
            raise Exception("Model is not trained yet")
        return self._alpha

    @property
    def dataset(self):
        return self.kernel.dataset

    @property
    def data(self):
        return self.kernel.data

    @property
    def n(self):
        return self.kernel.n

    @property
    def m(self):
        return self.kernel.m
