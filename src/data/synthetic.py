from src.data.dataset import Dataset
from src.tools.utils import Logger
import src.config as conf
import numpy as np


def gen_class_data(n, m, mode="gauss", nclasses=2):
    if mode == "gauss":
        data1 = np.random.normal(
            2 * np.ones(m), scale=np.arange(1, m + 1), size=(n // 2, m))
        data2 = np.random.normal(
            2 * np.ones(m), scale=np.ones(m), size=(n // 2, m))

        shuffle = np.arange(n)
        np.random.shuffle(shuffle)

        data = np.concatenate((data1, data2))[shuffle]
        labels = np.concatenate((np.ones(n // 2, dtype=int),
                                 np.zeros(n - n // 2, dtype=int)))[shuffle]
    elif mode == "circle":
        npp = n // nclasses
        angles = [
            np.random.rand(npp) * 2 * np.pi for _ in range(nclasses)
        ]
        radus = np.array([1, 5, 10])[:nclasses]
        radus = np.array([
            np.random.normal(x, scale=0.3, size=npp) for x in radus
        ])

        data = np.array([
            np.transpose([radu * np.cos(angle), radu * np.sin(angle)])
            for angle, radu in zip(angles, radus)
        ])
        data = np.concatenate(data)

        shuffle = np.arange(npp * nclasses)
        np.random.shuffle(shuffle)
        data = data[shuffle]
        labels = np.concatenate(
            (np.zeros(n // nclasses,
                      dtype=int), 1 * np.ones(n // nclasses, dtype=int),
             2 * np.ones(n // nclasses, dtype=int)))[shuffle]

    return (data, labels, None)


def gen_reg_data(n, m):
    shuffle = np.arange(n)
    np.random.shuffle(shuffle)
    
    x = np.linspace(0, 15, n)
    y = 2 * np.cos(x) + np.random.normal(size=n)

    data = x[shuffle].reshape(-1, 1)
    labels = y[shuffle]

    return data, labels


class GenClassData(Dataset):
    def __init__(self,
                 n,
                 m,
                 nclasses=2,
                 split_val=0.1,
                 seed=conf.SEED,
                 verbose=True,
                 mode="gauss"):

        data = gen_class_data(n, m, mode=mode, nclasses=nclasses)
        self.nclasses = nclasses

        super(GenClassData, self).__init__(
            *data, shuffle=True, seed=conf.SEED, verbose=verbose)

        self._log("Data class generated")

        self.conf = {
            "n": n,
            "m": m,
            "split_val": split_val,
            "nclasses": nclasses
        }

    def show(self):
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

    def show_class(self, predict):
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
        axpred.set_title("Prediction")
        axgt.set_title("Ground truth")
        plt.show()

    def show_pca2(self, proj, anddata=False):
        import matplotlib.pyplot as plt
        if anddata:
            f, (axdata, axpca) = plt.subplots(1, 2)

            if self.nclasses == 2:
                it = [-1, 1]
            else:
                it = range(self.nclasses)

            for i in it:
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


class GenRegData(Logger):
    def __init__(self,
                 n,
                 m,
                 seed=conf.SEED,
                 verbose=True):

        self.verbose = verbose
        self.data, self.labels = gen_reg_data(n, m)

        self._log("Data class generated")

        self.verbose = verbose
        self._log("Generated data..")
        
    def show(self):
        import matplotlib.pyplot as plt
        plt.scatter(self.data, self.labels)
        plt.title("Generated data")
        plt.show()

    def show_reg(self, reg):
        import matplotlib.pyplot as plt
        plt.scatter(self.data, self.labels)
        plt.scatter(self.data, np.array([reg(x[0]) for x in self.data]))
        plt.title("Regression")
        plt.show()


if __name__ == '__main__':
    # data = GenClassData(n=300, m=2)
    data = GenClassData(n=300, m=2, mode="gauss")
