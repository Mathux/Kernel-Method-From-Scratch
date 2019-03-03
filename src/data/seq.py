import src.config as conf
from src.config import SEED
from src.data.dataset import Dataset
from src.tools.utils import Logger


def SeqData(k=0,
            mat=False,
            small=False,
            verbose=True,
            shuffle=True,
            dataname="train",
            nsmall=200):
    assert (dataname in ["train", "test"])

    if dataname == "train":
        data, names = load_data(
            "train", k=k, mat=mat, small=small, nsmall=nsmall, givename=True)
        Logger.log(
            verbose,
            "Train data loaded! (" + names[0] + " and " + names[1] + ")")
    elif dataname == "test":
        data, names = load_data(
            "test", k=k, mat=mat, small=small, nsmall=nsmall, givename=True)
        Logger.log(verbose, "Test data loaded! (" + names + ")")

    dataset = Dataset(
        *data,
        shuffle=shuffle,
        seed=SEED,
        verbose=verbose,
        labels_change=True,
        name="seq")
    return dataset


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
            return (data, labels, Id), (datafilename, labelfilename)
        else:
            return (data, labels, Id)
    else:
        if givename:
            return (data, None, Id), datafilename
        else:
            return (data, None, Id)


if __name__ == '__main__':
    data = SeqData(k=0, dataname="train", mat=False, small=False, verbose=True)
