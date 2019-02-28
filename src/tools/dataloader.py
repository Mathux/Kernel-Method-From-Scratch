import numpy as np
import matplotlib.pyplot as plt
import src.config as conf
from src.tools.utils import sigmoid, Logger


class Dataset:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels


class GenClassData(Logger):
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


class GenRegData(Logger):
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


class SeqData(Logger):
    def __init__(self, small=False, verbose=True):
        from src.tools.utils import load_train, load_test
        self.verbose = verbose
        
        self._log("Load train data..")
        x_train, y_train = load_train()
        self._log("Train data loaded!")

        def transform(x, y):
            X = x[0]["seq"].values
            Y = y[0]["Bound"].values
            if small:
                X = X[:100]
                Y = Y[:100]
            return Dataset(X, Y)

        self.train = transform(x_train, y_train)
        self._log("Load test data..")
        x_test = load_test()
        self._log("Test data loaded!")
        
        self.test = Dataset(x_test, None)

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


if __name__ == '__main__':
    # data = SeqData()
    data = GenClassData(1200, 3, nclasses=2, mode="gauss")
