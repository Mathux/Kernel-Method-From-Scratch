import numpy as np
from src.methods.KMethod import KMethod


class KPCA(KMethod):
    def __init__(self, kernel, dim=2, verbose=True):
        super(KPCA, self).__init__(kernel=kernel, name="KPCA", verbose=verbose)
        self.dim = dim
        self._projections = None

    def project(self, dataset=None, labels=None):
        self.load_dataset(dataset, labels)
        
        if self._projections is not None:
            return self._projections

        K = self.kernel.KC
        self._log("Fitting kernel PCA..")
        self._log("Computing eigenvalues of centered K..")
        delta, U = np.linalg.eig(K)
        self._log("Eigenvalues computed..")
        projections = [
            K.dot(U.T[i] / np.sqrt(delta[i])) for i in range(self.dim)
        ]
        self._log("Fitting done!")

        self._projections = np.array(projections).T
        return self._projections

    @property
    def projections(self):
        if self._projections is None:
            self.project()
        return self._projections
        

if __name__ == "__main__":
    from src.data.synthetic import GenClassData
    data = GenClassData(500, 2, mode="circle")

    from src.kernels.quad import QuadKernel
    kernel = QuadKernel(data)

    DIM = 3
    kpca = KPCA(kernel, dim=DIM)
    proj = kpca.project()
    
    data.show_pca(proj, dim=DIM)
