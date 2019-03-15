import numpy as np
from src.tools.utils import sigmoid
from src.methods.KMethod import KMethod, KMethodCreate, klogger
from src.methods.wkrr import WKRR


class KLR(KMethod, metaclass=KMethodCreate):
    name = "klr"
    defaultParameters = {"lam": 1, "n_iter": 10, "tol": 10**-6, "eps": 10**-6}

    @klogger("kernel Logistic Regression")
    def fit(self, K):
        n = self.n
        Y = self.labels

        alpha_old = np.zeros(n)
        self._alpha = np.zeros(n)

        for i in self.vrange(self.param.n_iter):
            M = K.dot(self._alpha)
            sig_pos, sig_neg = sigmoid(M * Y), sigmoid(-M * Y)
            W = sig_neg * sig_pos
            Z = M + Y / np.maximum(sig_pos, self.param.eps)
            alpha_old = self._alpha
            wkrr = WKRR(
                kernel=self.kernel,
                parameters={"lam": self.param.lam},
                verbose=False)
            self._alpha = wkrr.fit(w=W, labels=Z)
            if np.linalg.norm(self._alpha - alpha_old) < self.param.tol:
                break
            if i == self.param.n_iter - 1:
                print("WARNING: KLR didn't converge")

        return self._alpha


if __name__ == "__main__":
    from src.tools.test import EasyTest
    dparams = {"small": True, "nsmall": 300}
    EasyTest(kernel="spectral", data="seq", method="klr", dparams=dparams)
