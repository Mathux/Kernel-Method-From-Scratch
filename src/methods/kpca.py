import numpy as np
from src.methods.KMethod import KMethod, KMethodCreate
from src.tools.utils import Logger


class KPCA(KMethod, metaclass=KMethodCreate):
    name = "kpca"
    defaultParameters = {"dim": 3}

    def project(self, dataset=None, labels=None):
        self._log("Fitting kernel PCA..")
        Logger.indent()
        self.load_dataset(dataset, labels)
        
        if self._projections is not None:
            return self._projections

        K = self.kernel.KC
        self._log("Computing eigenvalues of centered K..")
        delta, U = np.linalg.eig(K)
        self._log("Eigenvalues computed..")
        projections = [
            K.dot(U.T[i] / np.sqrt(delta[i])) for i in range(self.param.dim)
        ]
        Logger.dindent()
        self._log("Fitting done!\n")

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
