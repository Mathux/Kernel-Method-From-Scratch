from src.kernels.kernel import DataKernel, KernelCreate
from numpy import tanh


class SigmoidKernel(DataKernel, metaclass=KernelCreate):
    defaultParameters = {"offset": 0, "alpha": 1}

    def kernel(self, x, y):
        return tanh(self.param.alpha * x.dot(y) + self.param.offset)


if __name__ == "__main__":
    from src.data.synthetic import GenClassData
    data = GenClassData(300, 3, nclasses=2)
    kernel = SigmoidKernel(data)

    from src.methods.kpca import KPCA
    kpca = KPCA(kernel)
    proj = kpca.project()

    data.show_pca(proj)
