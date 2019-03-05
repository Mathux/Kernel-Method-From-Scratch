from src.tools.utils import Logger, Parameters
import numpy as np
from src.tools.utils import sigmoid


class KMethodCreate(type):
    def __init__(cls, clsname, superclasses, attributedict):
        def init(self, kernel=None, parameters=None, verbose=True):
            super(cls, self).__init__(
                kernel=kernel,
                name=clsname,
                parameters=parameters,
                verbose=verbose,
                cls=cls)

        cls.__init__ = init


class KMethod(Logger):
    def __init__(self,
                 kernel,
                 name="KMethod",
                 parameters=None,
                 verbose=True,
                 cls=None):

        self.verbose = verbose
        self.kernel = kernel

        self.param = Parameters(parameters, cls.defaultParameters)

        self.__name__ = name
        self.verbose = verbose
        self._alpha = None
        self.kernel = kernel
        self._labels = None

        # For the KPCA
        self._projections = None

    # Load the dataset (if there are one) in the kernel
    # or just load the labels
    def load_dataset(self, dataset=None, labels=None):
        if dataset is None:
            if labels is None:
                self._log("Taking data from the kernel directly")
                self._labels = self.kernel.labels
            else:
                self._labels = labels
        else:
            self._log("Load the data in the kernel")
            self.kernel.dataset = dataset
            self._labels = None  # take the kernel one

    def predict(self, x):
        K_xi = self.kernel.predict(x)
        return self.alpha.dot(K_xi)

    def predictBin(self, x):
        pred = self.predict(x)
        return 1 if pred > 0 else -1

    def predict_array(self, X, binaire=True):
        if binaire:
            fonc = self.predictBin
        else:
            fonc = self.predict

        return np.array([fonc(x) for x in X])

    def score_recall_precision(self, dataset):
        X, y = dataset.data, dataset.labels

        predictions = self.predict_array(X, binaire=True)

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
    def labels(self):
        if self._labels is None:
            return self.kernel.labels
        return self._labels

    @property
    def n(self):
        return self.kernel.n

    @property
    def m(self):
        return self.kernel.m

    def __str__(self):
        return self.__name__

    def __repr__(self):
        return self.__name__


def AllClassMethods():
    from src.methods.kknn import KKNN
    from src.methods.klr import KLR
    from src.methods.ksvm import KSVM

    return [KKNN, KLR, KSVM], ["kknn", "klr", "ksvm"]
