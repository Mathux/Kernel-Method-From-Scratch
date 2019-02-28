import numpy as np
from src.methods.KMethod import KMethod


class WKRR(KMethod):
    def __init__(self, kernel, lam=0.1, name="WKRR", verbose=False):
        super(WKRR, self).__init__(kernel=kernel, name=name, verbose=verbose)
        self.lam = lam

    def fit(self, dataset=None, labels=None, w=None):
        self.load_dataset(dataset, labels)
        n = self.n
        if w is None:
            w = np.ones(n)
        # Gram matrix
        K = self.kernel.K
        
        self._log("Fitting weighted kernel ridge regression...")
        # Prepare data
        W_half = np.diag(np.sqrt(w))

        inv = np.linalg.inv(
            W_half.dot(K.dot(W_half)) + n * self.lam * np.eye(n))
        # Solve for alpha
        self._alpha = W_half.dot(inv.dot(W_half.dot(self.labels)))
        self._log("Fitting done!")
        return self._alpha


class KRR(WKRR):
    def __init__(self, kernel, verbose=True):
        super(KRR, self).__init__(
            kernel=kernel, name="KRR", verbose=verbose)


if __name__ == "__main__":
    from src.tools.dataloader import GenRegData
    data = GenRegData(500, 2)

    # from kernel import LinearKernel
    from src.kernels.gaussian import GaussianKernel
    kernel = GaussianKernel(data.train)
    klr = KRR(kernel)
    klr.fit()
    data.show_reg(klr.predict)
