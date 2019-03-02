import numpy as np
from src.config import SEED
from src.tools.utils import Logger


class Dataset(Logger):
    def __init__(self,
                 data,
                 labels,
                 Id=None,
                 shuffle=True,
                 seed=SEED,
                 verbose=True,
                 labels_change=True):
        self.verbose = verbose
        self.data = data
        self.labels = labels
        self.Id = Id
        
        self.n = data.shape[0]
        self.m = data.shape[1] if len(data.shape) > 1 else None
        # Shuffle the dataset
        if shuffle:
            self._log("Dataset shuffled")
            self.shuffle(seed)
        if labels_change and labels is not None:
            self.transform_label()
    
    # Shuffle the dataset
    def shuffle(self, seed=SEED):
        np.random.seed(seed)
        mask = np.arange(self.n)
        np.random.shuffle(mask)
        # Shuffle elements
        self.iter(lambda x: x[mask])

    # Def split dataset
    def split(self, split_val=0.1, seed=SEED):
        self.shuffle(seed)
        value_split = int(split_val * self.n)
        train = self.map(lambda x: x[value_split:])
        val = self.map(lambda x: x[:value_split])
        self._log("Dataset splitted into train and val")
        return train, val

    # Def split k fold
    def kfolds(self, k=5, seed=SEED):
        self.shuffle(seed)
        nbF = self.n // k
        folds = list(
            map(lambda i: self.map(lambda x: x[i * nbF:(i + 1) * nbF]),
                range(k)))
        self._log("Dataset splitted into " + str(k) + " folds")
        return folds

    ITERON = ["data", "labels", "Id"]

    # apply a fonction to data/labels/Id in place
    def iter(self, func):
        dic = self.__dict__

        def f(data):
            if dic[data] is not None:
                dic[data] = func(dic[data])

        list(map(f, Dataset.ITERON))

    # apply a fonction to data/labels/Id and return a new dataset
    def map(self, func):
        dic = self.__dict__

        def f(data):
            return func(dic[data]) if dic[data] is not None else None

        return Dataset(*map(f, Dataset.ITERON))

    # To add two Dataset together
    def __add__(self, other):
        assert (type(other) == Dataset)
        data = np.concatenate((self.data, other.data))
        labels = np.concatenate((self.labels, other.labels))
        if self.Id is not None and other.Id is not None:
            Id = np.concatenate((self.Id, other.Id))
        else:
            Id = None
        return Dataset(data, labels, Id)

    # Invert labels signification
    def transform_label(self, inplace=True):
        def replace(v1, v2, y):
            if not inplace:
                y = np.copy(y)
            y[y == v1] = v2
            return y

        y = self.labels
        if -1 in y:
            y = replace(-1, 0, y)
        elif 0 in y:
            y = replace(0, -1, y)
        else:
            raise Exception("Bad labels")

    # Show project of data
    def show_pca(self, proj, dim):
        import matplotlib.pyplot as plt
        proj = proj.real
        if self.nclasses == 2:
            it = [-1, 1]
        else:
            it = range(self.nclasses)
        if dim == 2:
            for i in it:
                mask = self.labels == i
                plt.scatter(proj[mask][:, 0], proj[mask][:, 1])
        elif dim == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for i in it:
                mask = self.labels == i
                ax.scatter(proj[mask][:, 0], proj[mask][:, 1],
                           proj[mask][:, 2])

        plt.title("KPCA")
        plt.show()

    # Invert labels signification
    def __invert__(self):
        # if no label, just quit
        if self.labels is None:
            raise Exception("Can't revert empty labels")
        labels = self.transform_label(inplace=False)
        # Not very good: cls can be deferent: todo
        return Dataset(self.data, labels, self.Id)

    def __str__(self):
        size = "(" + str(self.n) + ", " + str(self.m) + ")"
        return "Dataset object of size " + size
