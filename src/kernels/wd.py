import numpy as np
from src.kernels.kernel import Kernel


class WDKernel(Kernel):
    def __init__(self, dataset=None, d=3, verbose=True):
        super(WDKernel, self).__init__(
            dataset=dataset, name="WDKernel", verbose=verbose)
        self.d = d
        self.beta = 2 * np.linspace(1, d, d, dtype='int')[::-1] / (d * (d + 1))

    def kernel(self, x, y):
        k_xy = 0
        assert len(x) == len(y)
        lenght = len(x)
        for k in range(self.d):
            temp = 0
            for i in range(lenght - k):
                temp += 1 * (x[i:i + k] == y[i:i + k])
            k_xy += self.beta[k] * temp
        return k_xy


if __name__ == "__main__":
    from src.tools.dataloader import SeqData
    data = SeqData()
    kernel = WDKernel(data.train, d=3, verbose=True)
    K = kernel.K
