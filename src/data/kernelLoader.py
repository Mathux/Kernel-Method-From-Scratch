# Ugly but necessary for pickle
from src.kernels.spectral import SpectralKernel
from src.kernels.mismatch import MismatchKernel
from src.kernels.wd import WDKernel
from src.kernels.la import LAKernel
from src.kernels.exponential import ExponentialKernel
from src.kernels.gaussian import GaussianKernel
from src.kernels.laplacian import LaplacianKernel
from src.kernels.linear import LinearKernel
from src.kernels.polynomial import PolynomialKernel
from src.kernels.quad import QuadKernel
from src.kernels.sigmoid import SigmoidKernel

import pickle 
from src.config import kernelSavePath, kernelSaveExt
from os.path import join as pjoin
from src.tools.utils import create_dir


def load(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save(kernel, path=None):
    dataName = kernel.__name__
    params = kernel.param.dic
    for key, values in params.items() :
        dataName += '_' + str(key) + '_' + str(values)
    dataName = dataName + '_' + kernel.dataset.__name__ + kernelSaveExt
    
    if path is None:
        create_dir(kernelSavePath)
        path = pjoin(kernelSavePath, dataName)
    else:
        create_dir(path)
        path = pjoin(path, "kernel" + kernelSaveExt)
        
    kernel._log("The kernel in saved in " + path)
    with open(path, 'wb') as handle:
        pickle.dump(kernel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return path
