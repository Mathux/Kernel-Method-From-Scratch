from src.data.dataset import Dataset
from src.tools.utils import Logger, Parameters
import numpy as np


class __GenClassData:
    def __init__(self):
        self.defaultParameters = {
            "n": 500,
            "m": 2,
            "nclasses": 2,
            "split_val": 0.1,
            "mode": "circle",
            "shuffle": False,
            "labels_change": True,
            "name": "synth"
        }
        self.name = "synth"
        
    def __call__(self, parameters=None, verbose=True):
        Logger.log(verbose, "Loading datasets...")
        Logger.indent()
        p = Parameters(parameters, self.defaultParameters)
        dataset = {}
        for nameset in ["train", "test"]:
            data = gen_class_data(p.n, p.m, mode=p.mode, nclasses=p.nclasses)
            Logger.log(verbose, "synthetic " + nameset + " data generated")
            dataset[nameset] = Dataset(p, *data, verbose=verbose)

        Logger.dindent()
        Logger.log(verbose, "datasets loaded!\n")
        return [dataset]


GenClassData = __GenClassData()


class __GenRegData:
    def __init__(self):
        self.defaultParameters = {
            "n": 300,
            "m": 2,
            "nclasses": 1,
            "shuffle": True,
            "name": "synth"
        }
        self.name = "synth"
        
    def __call__(self, parameters=None, verbose=True):
        Logger.log(verbose, "Loading datasets...")
        Logger.indent()
        p = Parameters(parameters, self.defaultParameters)        
        dataset = {}
        for nameset in ["train", "test"]:
            data, labels = gen_reg_data(p.n, p.m)
            Logger.log(verbose, "synthetic " + nameset + " data generated")            
            dataset[nameset] = Dataset(p, data, verbose=verbose)
            
        Logger.dindent()
        Logger.log(verbose, "datasets loaded!\n")
        return [dataset]


GenRegData = __GenRegData()


def gen_class_data(n, m, mode="circle", nclasses=2):
    assert (nclasses == 2)
    if mode == "gauss":
        data1 = np.random.normal(
            2 * np.ones(m), scale=np.arange(1, m + 1), size=(n // 2, m))
        data2 = np.random.normal(
            2 * np.ones(m), scale=np.ones(m), size=(n // 2, m))

        shuffle = np.arange(n)
        np.random.shuffle(shuffle)

        data = np.concatenate((data1, data2))[shuffle]
        labels = np.concatenate((np.zeros(n // 2, dtype=int),
                                 np.ones(n - n // 2, dtype=int)))[shuffle]
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
        labels = np.concatenate((np.zeros(n // 2, dtype=int),
                                 np.ones(n - n // 2, dtype=int)))[shuffle]

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
    data = GenClassData()
