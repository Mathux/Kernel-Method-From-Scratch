import numpy as np
from src.kernels.kernel import DataKernel, KernelCreate


class WDKernel(DataKernel, metaclass=KernelCreate):
    defaultParameters = {"d": 3}

    def _kernel(self, x, y):
        beta = 2 * np.linspace(
            1, self.param.d, self.param.d,
            dtype='int')[::-1] / (self.param.d * (self.param.d + 1))
        k_xy = 0
        assert len(x) == len(y)
        lenght = len(x)
        for k in range(self.param.d):
            temp = 0
            for i in range(lenght - k):
                temp += 1 * (x[i:i + k] == y[i:i + k])
            k_xy += beta[k] * temp
        return k_xy


if __name__ == "__main__":
    from src.data.seq import SeqData
    data = SeqData(small=True, nsmall=500)
        
    kernel = WDKernel(data)

    from src.methods.kpca import KPCA
    kpca = KPCA(kernel)
    proj = kpca.project()

    data.show_pca(proj)
