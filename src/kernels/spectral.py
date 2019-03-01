import numpy as np
from src.kernels.kernel import StringKernel, EasyCreate


class SpectralKernel(StringKernel, metaclass=EasyCreate):
    defaultParameters = {"k": 3}

    def _compute_phi(self, x):
        phi = np.zeros(len(self.mers))
        for j, mer in enumerate(self.mers):
            phi[j] += 1 * (mer in x)
        return phi


if __name__ == "__main__":
    from src.data.seq import SeqData
    data = SeqData(small=True)
    kernel = SpectralKernel(data)
    K = kernel.K
