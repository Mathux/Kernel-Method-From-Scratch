import numpy as np
from src.tools.utils import sigmoid
from src.methods.KMethod import KMethod
from src.methods.wkrr import WKRR


class KLR(KMethod):
    def __init__(self, kernel, lam=1, n_iter=100, verbose=True):
        super(KLR, self).__init__(kernel=kernel, name="KLR", verbose=verbose)
        self.lam = lam
        self.n_iter = n_iter

    def fit(self, dataset=None, labels=None, tol=10**-5, eps=10**-5):
        # Load the dataset (if there are one) in the kernel
        self.load_dataset(dataset, labels)

        n = self.n
        Y = self.labels
        # K is computed internally here if new data loaded
        K = self.kernel.K
        alpha_old = np.zeros(n)
        self._alpha = np.zeros(n)

        self._log("Fitting kernel logistic regression..")

        for i in self.vrange(self.n_iter):
            M = K.dot(self._alpha)
            sig_pos, sig_neg = sigmoid(M * Y), sigmoid(-M * Y)
            W = sig_neg * sig_pos
            Z = M + Y / np.maximum(sig_pos, eps)
            alpha_old = self._alpha
            self._alpha = WKRR(
                kernel=self.kernel, lam=self.lam).fit(w=W, labels=Z)
            if np.linalg.norm(self._alpha - alpha_old) < tol:
                break
            if i == self.n_iter - 1:
                print("WARNING: KLR didn't converge")

        self._log("Fitting done!")
        return self._alpha


if __name__ == "__main__":
    from src.tools.dataloader import GenClassData
    data = GenClassData(500, 2)

    from src.kernels.linear import LinearKernel
    kernel = LinearKernel(data.train)
    klr = KLR(kernel)
    klr.fit()

    data.show_class(klr.predict)
