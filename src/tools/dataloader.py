import numpy as np
import matplotlib.pyplot as plt
import src.config as conf
from src.tools.utils import sigmoid, Logger


class Dataset:
    def __init__(self, data, labels, Id=None):
        self.data = data
        self.labels = labels
        self.Id = Id


class Data(Logger):
    def __init__(self):
        return

    def transform_label(y):
        if -1 in y:
            y[y == -1] = 1
        elif 0 in y:
            y[y == 0] = -1
        else:
            raise Exception('Bad labels')
        return y  # for convenience
    
    
class GenClassData(Data):
    def __init__(self,
                 n,
                 m,
                 nclasses=2,
                 split_val=0.1,
                 seed=conf.SEED,
                 verbose=True,
                 small=False,
                 mode="gauss"):

        self.verbose = verbose
        self._log("Generated data..")

        self.n = n
        self.m = m
        self.nclasses = nclasses

        if mode == "gauss":
            assert self.nclasses == 2
            data1 = np.random.normal(
                2 * np.ones(m), scale=np.arange(1, m + 1), size=(n // 2, m))
            data2 = np.random.normal(
                2 * np.ones(m), scale=np.ones(m), size=(n // 2, m))

            shuffle = np.arange(n)
            np.random.shuffle(shuffle)

            self.data = np.concatenate((data1, data2))[shuffle]
            self.labels = np.concatenate((np.ones(n // 2, dtype=int),
                                          np.zeros(n - n // 2,
                                                   dtype=int)))[shuffle]

        elif mode == "circle":
            assert self.nclasses == 3
            angles = [np.random.rand(n // 3) * 2 * np.pi for _ in range(3)]
            radus = [1, 5, 10]
            radus = np.array(
                [np.random.normal(x, scale=0.3, size=n // 3) for x in radus])

            data = np.array([
                np.transpose([radu * np.cos(angle), radu * np.sin(angle)])
                for angle, radu in zip(angles, radus)
            ])
            data = np.concatenate(data)

            shuffle = np.arange(n)
            np.random.shuffle(shuffle)

            self.data = data[shuffle]
            self.labels = np.concatenate(
                (np.zeros(n // 3, dtype=int), 1 * np.ones(n // 3, dtype=int),
                 2 * np.ones(n // 3, dtype=int)))[shuffle]

        self.train = Dataset(self.data[int(split_val * n):],
                             self.labels[int(split_val * n):])
        self.val = Dataset(self.data[:int(split_val * n)],
                           self.labels[:int(split_val * n)])

        self.conf = {
            "n": n,
            "m": m,
            "split_val": split_val,
            "nclasses": nclasses
        }

        self._log("Data generated!")

    def show(self):
        for i in range(self.nclasses):
            mask = self.labels == i
            plt.scatter(self.data[mask][:, 0], self.data[mask][:, 1])
        plt.title("Generated data")
        plt.show()

    def show_class(self, predict):
        def clas(x):
            return 1 if sigmoid(predict(x)) >= 0.5 else 0

        f, (axgt, axpred) = plt.subplots(1, 2)
        for i in range(self.nclasses):
            mask_pred = np.array([clas(x) == i for x in self.data])
            mask_gt = self.labels == i
            axpred.scatter(self.data[mask_pred][:, 0],
                           self.data[mask_pred][:, 1])
            axgt.scatter(self.data[mask_gt][:, 0], self.data[mask_gt][:, 1])
        axpred.set_title("Prediction")
        axgt.set_title("Ground truth")
        plt.show()

    def show_pca(self, proj, anddata=False):
        if anddata:
            f, (axdata, axpca) = plt.subplots(1, 2)

            for i in range(self.nclasses):
                mask = self.labels == i
                axdata.scatter(self.data[mask][:, 0], self.data[mask][:, 1])
                axpca.scatter(proj[mask][:, 0], proj[mask][:, 1])

            axdata.set_title("Data")
            axpca.set_title("KPCA")
            plt.show()

        else:
            for i in range(self.nclasses):
                mask = self.labels == i
                plt.scatter(proj[mask][:, 0], proj[mask][:, 1])
            plt.title("PCA on generated data")
            plt.show()


class GenRegData(Data):
    def __init__(self,
                 n,
                 m,
                 nclasses=2,
                 split_val=0.1,
                 seed=conf.SEED,
                 verbose=True,
                 small=False):

        self._log("Generated data..")

        self.n = n
        self.m = m

        shuffle = np.arange(n)
        np.random.shuffle(shuffle)

        x = np.linspace(0, 15, n)
        y = 2 * np.cos(x) + np.random.normal(size=n)

        self.data = x[shuffle].reshape(-1, 1)
        self.labels = y[shuffle]

        self.train = Dataset(self.data[int(split_val * n):],
                             self.labels[int(split_val * n):])
        self.val = Dataset(self.data[:int(split_val * n)],
                           self.labels[:int(split_val * n)])

        self.config = {
            "n": n,
            "m": m,
            "split_val": split_val,
            "nclasses": nclasses
        }

        self._log("Data generated!")

    def show(self):
        plt.scatter(self.data, self.labels)
        plt.title("Generated data")
        plt.show()

    def show_reg(self, reg):
        plt.scatter(self.data, self.labels)
        plt.scatter(self.data, np.array([reg(x[0]) for x in self.data]))
        plt.title("Regression")
        plt.show()


class SeqData(Data):
    def __init__(self, k=0, mat=False, small=False, verbose=True):
        self.verbose = verbose

        self._log("Load train data (k=" + str(k) + ")")
        self.train, names = load_data(
            "train", k=k, mat=mat, small=small, givename=True)
        self._log("Train data loaded! (" + names[0] + " and " + names[1] + ")")

        self._log("Load test data (k=" + str(k) + ")")
        self.test, names = load_data(
            "test", k=k, mat=mat, small=small, givename=True)
        self._log("Test data loaded! (" + names + ")")

        self.nclasses = 2

    def show_pca(self, proj, dim):
        import matplotlib.pyplot as plt
        proj = proj.real
        if dim == 2:
            for i in range(self.nclasses):
                mask = self.train.labels == i
                plt.scatter(proj[mask][:, 0], proj[mask][:, 1])
        elif dim == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for i in range(self.nclasses):
                mask = self.train.labels == i
                ax.scatter(proj[mask][:, 0], proj[mask][:, 1],
                           proj[mask][:, 2])

        plt.title("PCA on seq data")
        plt.show()


# Loading data
def load_data(name, k=0, mat=False, small=False, nsmall=100, givename=False):
    from os.path import join as pjoin
    import pandas as pd
    st = "tr" if name == "train" else "te" if name == "test" else None
    assert (st is not None)

    datafilename = "X" + st + str(k) + conf.ext
    dataPath = pjoin(conf.dataPath, datafilename)
    data = pd.read_csv(dataPath, sep=',')

    def shrink(x):
        return x[:nsmall] if small else x

    Id = shrink(data["Id"])

    if mat:
        datafilename = "X" + st + str(k) + "_mat100" + conf.ext
        dataPath = pjoin(conf.dataPath, datafilename)

        datamat = pd.read_csv(dataPath, sep=' ', dtype='float64', header=None)
        data = datamat.values
    else:
        data = data["seq"].values

    data = shrink(data)

    if name == "train":
        labelfilename = "Y" + st + str(k) + conf.ext
        labelPath = pjoin(conf.dataPath, labelfilename)
        labels = pd.read_csv(labelPath)
        labels = shrink(labels["Bound"].values)
        # convert
        Data.transform_label(labels)
        if givename:
            return Dataset(data, labels, Id), (datafilename, labelfilename)
        else:
            return Dataset(data, labels, Id)
    else:
        if givename:
            return data, datafilename
        else:
            return Dataset(data, None, Id)


if __name__ == '__main__':
    data = SeqData(k=0, mat=False, small=False, verbose=True)
