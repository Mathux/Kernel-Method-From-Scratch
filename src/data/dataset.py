import numpy as np
from src.config import SEED
from src.tools.utils import Logger


class Dataset(Logger):
    def __init__(self,
                 data,
                 labels,
                 Id=None,
                 seed=SEED,
                 verbose=True,
                 labels_change=False,
                 shuffle=False,
                 nclasses=2,
                 name="Dataset"):
        self.nclasses = nclasses
        self.__name__ = name
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
        value_split = int(split_val * self.n)
        train = self.map(lambda x: x[value_split:])
        val = self.map(lambda x: x[:value_split])
        self._log("Dataset splitted into train and val")
        return train, val

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
        assert (self.__name__ == other.__name__)
        name = self.__name__
        data = np.concatenate((self.data, other.data))
        labels = np.concatenate((self.labels, other.labels))
        if self.Id is not None and other.Id is not None:
            Id = np.concatenate((self.Id, other.Id))
        else:
            Id = None
        return Dataset(data, labels, Id, name=name)

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
    def show_pca(self, proj, predict=None, dim=3):
        import matplotlib.pyplot as plt
        if predict is not None:

            predictions = np.array([predict(x) for x in self.data])

            if dim == 2:
                fig, (axgt, axpred) = plt.subplots(1, 2)

            elif dim == 3:
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure()
                axgt = fig.add_subplot(121, projection='3d')
                axpred = fig.add_subplot(122, projection='3d')

            axgt.set_title("Ground truth")
            axpred.set_title("Prediction")
            axes = [axgt, axpred]
            labels = [self.labels, predictions]

        else:
            if dim == 2:
                fig, axgt = plt.subplots(1, 1)

            elif dim == 3:
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure()
                axgt = fig.add_subplot(111, projection='3d')

            axgt.set_title("Ground truth")
            axes = [axgt]
            labels = [self.labels]

        proj = proj.real
        if self.nclasses == 2:
            it = [-1, 1]
        else:
            it = range(self.nclasses)

        def scatter(ax, mask):
            args = [proj[mask][:, i] for i in range(dim)]
            ax.scatter(*args)

        for i in it:
            for ax, label in zip(axes, labels):
                mask = label == i
                scatter(ax, mask)
        plt.show()

    def _show_gen_class_data(self):
        import matplotlib.pyplot as plt
        if self.nclasses == 2:
            it = [-1, 1]
        else:
            it = range(self.nclasses)
        for i in it:
            mask = self.labels == i
            plt.scatter(self.data[mask][:, 0], self.data[mask][:, 1])
        plt.title("Generated data")
        plt.show()

    def _show_gen_class_predicted(self, predict):
        def clas(x):
            return -1 if predict(x) < 0 else 1

        import matplotlib.pyplot as plt
        f, (axgt, axpred) = plt.subplots(1, 2)
        if self.nclasses == 2:
            it = [-1, 1]
        else:
            it = range(self.nclasses)
        for i in it:
            mask_pred = np.array([clas(x) == i for x in self.data])
            mask_gt = self.labels == i
            axpred.scatter(self.data[mask_pred][:, 0],
                           self.data[mask_pred][:, 1])
            axgt.scatter(self.data[mask_gt][:, 0], self.data[mask_gt][:, 1])
        plt.show()

    def _show_gen_reg_data(self):
        import matplotlib.pyplot as plt
        plt.scatter(self.data, self.labels)
        plt.title("Generated data")
        plt.show()

    def _show_gen_reg_predicted(self, reg):
        import matplotlib.pyplot as plt
        plt.scatter(self.data, self.labels)
        plt.scatter(self.data, np.array([reg(x[0]) for x in self.data]))
        plt.title("Regression")
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
        return "Dataset object of size " + size + " with " + self.__name__ + " data"

    def __repr__(self):
        return self.__str__()


class KFold(Logger):
    def __init__(self, dataset, kfold, verbose=True):
        self.verbose = verbose

        self.dataset = dataset
        self.kfold = kfold

        nbF = dataset.n // kfold
        folds = list(
            map(lambda i: self.dataset.map(lambda x: x[i * nbF:(i + 1) * nbF]),
                range(kfold)))
        self._log("Dataset splitted into " + str(kfold) + " folds")

        self.folds = folds

    def __getitem__(self, key):
        assert (key in range(self.kfold))
        return KFold.merge_folds(self.folds, key), self.folds[key]

    @staticmethod
    def merge_folds(folds, j):
        return np.sum([fold for i, fold in enumerate(folds) if not i == j])


def AllClassData():
    from src.data.seq import SeqData
    from src.data.synthetic import GenClassData

    return [SeqData, GenClassData], ["seq", "synth"]
