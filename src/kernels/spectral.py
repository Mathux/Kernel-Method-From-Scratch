import numpy as np
from src.kernels.kernel import StringKernel, KernelCreate


class SpectralKernel(StringKernel, metaclass=KernelCreate):
    defaultParameters = {"k": 6}

    def _compute_phi(self, x):
        phi = np.zeros(len(self.mers))
        for j, mer in enumerate(self.mers):
            phi[j] += 1 * (mer in x)
        return phi


if __name__ == "__main__":
    from src.data.seq import SeqData
    data = SeqData(small=True, nsmall=500)
        
    kernel = SpectralKernel(data, parameters = {'k' : 3, 'm' : 2})
    from src.methods.kpca import KPCA
    kpca = KPCA(kernel)
    proj = kpca.project()

    data.show_pca(proj)
