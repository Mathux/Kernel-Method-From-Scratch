import src.config as conf
from src.config import SEED
from src.data.dataset import Dataset


class SeqData(Dataset):
    def __init__(self,
                 k=0,
                 mat=False,
                 small=False,
                 verbose=True,
                 shuffle=True,
                 dataname="train"):
        assert (dataname in ["train", "test"])
        self.verbose = verbose
        
        if dataname == "train":
            self._log("Load train data (k=" + str(k) + ")")
            data, names = load_data(
                "train", k=k, mat=mat, small=small, givename=True)
            self._log("Train data loaded! (" + names[0] + " and " + names[1] +
                      ")")
        elif dataname == "test":
            self._log("Load test data (k=" + str(k) + ")")
            data, names = load_data(
                "test", k=k, mat=mat, small=small, givename=True)
            self._log("Test data loaded! (" + names + ")")

        self.nclasses = 2

        super(SeqData, self).__init__(
            *data, shuffle=shuffle, seed=SEED, verbose=verbose)

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
