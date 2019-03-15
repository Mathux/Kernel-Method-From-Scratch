import numpy as np
from src.methods.KMethod import KMethod, KMethodCreate, klogger


class KPCA(KMethod, metaclass=KMethodCreate):
    name = "kpca"
    defaultParameters = {"dim": 3}
    
    @klogger("Kernel Principal Component Analysis")
    def project(self, K):        
        if self._projections is not None:
            return self._projections

        self._log("Computing eigenvalues of centered K..")
        delta, U = np.linalg.eig(K)
        self._log("Eigenvalues computed..")
        projections = [
            K.dot(U.T[i] / np.sqrt(delta[i])) for i in range(self.param.dim)
        ]

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
    parameters = {"dim": DIM}
    kpca = KPCA(kernel, parameters=parameters)
    proj = kpca.project()

    data.show_pca(proj, dim=DIM)
