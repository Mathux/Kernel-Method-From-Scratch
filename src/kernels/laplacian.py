from src.kernels.kernel import DataKernel, KernelCreate
import numpy as np


class LaplacianKernel(DataKernel, metaclass=KernelCreate):
    defaultParameters = {"sigma": 1}

    def kernel(self, x, y):
        return np.exp(-np.linalg.norm(x-y)/self.param.sigma)


if __name__ == "__main__":
    from src.data.synthetic import GenClassData
    data = GenClassData(300, 3, nclasses=2)
    kernel = LaplacianKernel(data)

    from src.methods.kpca import KPCA
    kpca = KPCA(kernel)
    proj = kpca.project()

    data.show_pca(proj)
