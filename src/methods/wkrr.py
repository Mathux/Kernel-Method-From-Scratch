import numpy as np
from src.methods.KMethod import KMethod, KMethodCreate, klogger


class WKRR(KMethod, metaclass=KMethodCreate):
    defaultParameters = {"lam": 1}

    @klogger("Weighted Kernel Ridge Regression", wkrr=True)
    def fit(self, K, w):
        n = self.n
        if w is None:
            w = np.ones(n)

        # Prepare data
        W_half = np.diag(np.sqrt(w))

        inv = np.linalg.inv(
            W_half.dot(K.dot(W_half)) + n * self.param.lam * np.eye(n))
        # Solve for alpha
        self._alpha = W_half.dot(inv.dot(W_half.dot(self.labels)))
        return self._alpha


class KRR(WKRR):
    def __init__(self, kernel, verbose=True):
        super(KRR, self).__init__(
            kernel=kernel, name="KRR", verbose=verbose)


if __name__ == "__main__":
    from src.data.synthetic import GenRegData
    data = GenRegData(500, 2)

    from src.kernels.gaussian import GaussianKernel
    kernel = GaussianKernel(data)
    
    krr = KRR(kernel)
    krr.fit()

    data.show_reg(krr.predict)
