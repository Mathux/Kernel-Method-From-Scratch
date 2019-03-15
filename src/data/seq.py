import src.config as conf
from src.data.dataset import Dataset
from src.tools.utils import Parameters, Logger


class __SeqData:
    def __init__(self):
        self.defaultParameters = {
            "k": 0,
            "mat": False,
            "shuffle": False,
            "small": False,
            "nsmall": 200,
            "labels_change": True,
            "name": "seq",
            "shuffle": False,
            "nclasses": 2
        }
        self.name = "seq"
        
    def __call__(self, parameters=None, verbose=True):
        Logger.log(verbose, "Loading datasets...")
        Logger.indent()
        p = Parameters(parameters, self.defaultParameters)
        dataset = {}
        for nameset in ["train", "test"]:
            data, names = load_data(
                nameset,
                k=p.k,
                mat=p.mat,
                small=p.small,
                nsmall=p.nsmall,
                givename=True)
            names = "(" + " and ".join(names) + ")"
            Logger.log(verbose, nameset + " data loaded! " + names)

            dataset[nameset] = Dataset(p, *data, verbose=verbose)

        Logger.dindent()
        Logger.log(verbose, "datasets loaded!\n")
        return [dataset]


SeqData = __SeqData()


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
        if givename:
            return (data, labels, Id), [datafilename, labelfilename]
        else:
            return (data, labels, Id)
    else:
        if givename:
            return (data, None, Id), [datafilename]
        else:
            return (data, None, Id)


class __AllSeqData:
    def __init__(self):
        self.defaultParameters = {
            "k": 0,
            "shuffle": False,
            "small": False,
            "nsmall": 200,
            "labels_change": True,
            "name": "seq",
            "nclasses": 2
        }
        self.name = "allseq"
        
    def __call__(self, parameters=None, verbose=True):
        p = Parameters(parameters, self.defaultParameters)
        Logger.log(verbose, "Loading datasets...")
        Logger.indent()
        datasets = []
        for i in range(3):
            p.k = i
            dataset = SeqData(p, verbose=verbose)
            datasets.append(dataset[0])

        Logger.dindent()
        Logger.log(verbose, "datasets loaded!\n")
        return datasets


AllSeqData = __AllSeqData()

if __name__ == '__main__':
    # dataset = SeqData()
    datasets = AllSeqData()
