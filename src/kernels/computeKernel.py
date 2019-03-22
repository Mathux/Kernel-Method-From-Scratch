import os
from src.tools.test import findKernel, findData
from src.tools.utils import create_dir
from src.data.kernelLoader import save
from src.data.dataset import KFold


def cartesian_prods(paramsS):
    pools = []
    for params in paramsS:
        for key, vallist in params.items():
            pools.append([(key, val) for val in vallist])
                
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
        
    paramsF = []
    for res in result:
        paramsF.append(dict(res))
    return paramsF


KFOLDS = 4
DATA = "allseq"
ORIGIN = "computed"


# dparams = {"small": True, "nsmall": 10}

Datasets = findData(DATA)()  # dparams)
ndatas = len(Datasets)


KERNEL = "spectral"
# {"k": 5, "m": 1, 'la': 1, "trie": False}

# paramsS = [{"k": [6, 7, 8, 9, 10], "m": [1, 2, 3], "la": [0.]}]
paramsS = [{"k": [6, 7, 8]}]
paramsS = cartesian_prods(paramsS)
Kernel = findKernel(KERNEL)


for i, Dataset in enumerate(Datasets):
    pathDataset = os.path.join(ORIGIN, "dataset_" + str(i))
    train = Dataset["train"]
    folds = KFold(train, KFOLDS, verbose=False)
    for j in range(KFOLDS):
        pathFold = os.path.join(pathDataset, "fold_" + str(j))
        train, val = folds[j]
        pathKernel = os.path.join(pathFold, KERNEL)
        for params in paramsS:
            kernel = Kernel(train, parameters=params, verbose=False)
            _ = kernel.KC
            path = kernel.param.topath(pathKernel)
            save(kernel, path=kernel.param.topath(pathKernel))
