import numpy as np
from src.tools.utils import sigmoid
from src.methods.KMethod import KMethod, KMethodCreate
from src.methods.wkrr import WKRR


class KLR(KMethod, metaclass=KMethodCreate):
    defaultParameters = {"lam": 1, "n_iter": 10, "tol": 10**-5, "eps": 10**-5}

    def fit(self, dataset=None, labels=None):
        # Load the dataset (if there are one) in the kernel
        self.load_dataset(dataset, labels)

        n = self.n
        Y = self.labels
        # K is computed internally here if new data loaded
        K = self.kernel.K
        alpha_old = np.zeros(n)
        self._alpha = np.zeros(n)

        self._log("Fitting kernel logistic regression..")

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

        self._log("Fitting done!")
        return self._alpha


if __name__ == "__main__":
    from src.data.synthetic import GenClassData
    data = GenClassData(500, 2, mode="circle")

    from src.kernels.gaussian import GaussianKernel
    kernel = GaussianKernel(data)
    klr = KLR(kernel, parameters={"lam": 2})
    klr.fit()
    
    # data._show_gen_class_predicted(klr.predict)
