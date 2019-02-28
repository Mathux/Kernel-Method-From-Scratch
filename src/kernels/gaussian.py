import numpy as np
from src.kernels.kernel import Kernel


class GaussianKernel(Kernel):
    def __init__(self, dataset=None, sigma=1, verbose=True):
        super(GaussianKernel, self).__init__(
            dataset=dataset, name="gaussian", verbose=verbose)

        self.sigma = sigma

    def kernel(self, x, y):
        return np.exp(-np.linalg.norm(x - y)**2) / self.sigma**2


if __name__ == "__main__":
    from src.tools.dataloader import GenClassData
    data = GenClassData(300, 3, nclasses=2, mode="gauss")
    kernel = GaussianKernel(data.train, sigma=1, verbose=True)
    K = kernel.K
