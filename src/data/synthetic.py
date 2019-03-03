from src.data.dataset import Dataset
from src.tools.utils import Logger
import src.config as conf
import numpy as np


def GenClassData(n=500,
                 m=2,
                 nclasses=2,
                 split_val=0.1,
                 seed=conf.SEED,
                 verbose=True,
                 mode="circle"):
    data = gen_class_data(n, m, mode=mode, nclasses=nclasses)
    Logger.log(verbose, "Synthetic data class generated")
    dataset = Dataset(
        *data,
        shuffle=True,
        seed=conf.SEED,
        nclasses=nclasses,
        verbose=verbose,
        labels_change=True,
        name="synth")
    return dataset


def GenRegData(n, m, seed=conf.SEED, verbose=True):
    data, labels = gen_reg_data(n, m)
    Logger.log(verbose, "Synthetic data reg generated")
    dataset = Dataset(
        data,
        shuffle=True,
        seed=conf.SEED,
        nclasses=1,
        verbose=verbose,
        name="synth")
    return dataset


def gen_class_data(n, m, mode="circle", nclasses=2):
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
        angles = [np.random.rand(npp) * 2 * np.pi for _ in range(nclasses)]
        radus = np.array([1, 5, 10])[:nclasses]
        radus = np.array(
            [np.random.normal(x, scale=0.3, size=npp) for x in radus])

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


if __name__ == '__main__':
    data = GenClassData(n=300, m=2, mode="circle")
