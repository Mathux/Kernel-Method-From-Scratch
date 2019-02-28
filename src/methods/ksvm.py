from cvxopt import matrix, solvers
import numpy as np
from src.methods.KMethod import KMethod

# Solve the QP Problem:
#  minimize    1/2 x^T*P*x + q^T*x
#  subject to  G*x <= h
#              A*x <= b
# solvers.qp(P, q, G, h)

# Our SVM problem is:
#  minimize    (1/2)*x^T*K*x - y^T*x
#  subject to  0 <= yi*xi <= C
# The last inequaliy can be seen as:
#  diag(y)^T x <= (C, .., C)
# -diag(y)^T x <= (0, .., 0)


class KSVM(KMethod):
    def __init__(self, kernel, C=10**10, verbose=True):
        super(KSVM, self).__init__(kernel=kernel, name="KLR", verbose=verbose)
        self.C = C

    def fit(self, dataset=None, labels=None, verbose=False):
        # Load the dataset (if there are one) in the kernel
        self.load_dataset(dataset, labels)
        self._log("Fitting kernel svm..")

        n = self.n
        y = self.labels
        K = self.kernel.K
        C = self.C

        P = matrix(K, (n, n), "d")
        q = matrix(-y, (n, 1), "d")
        G = matrix(np.concatenate((np.diag(y), np.diag(-y))), (2 * n, n), "d")
        h = matrix(
            np.concatenate((C * np.ones(n), np.zeros(n))), (2 * n, 1), "d")
        self._alpha = np.array(solvers.qp(P, q, G, h)["x"]).reshape(-1)

        self._log("Fitting done!")
        return self._alpha


if __name__ == "__main__":
    from src.tools.dataloader import GenClassData
    data = GenClassData(100, 2)

    from src.kernels.gaussian import GaussianKernel
    kernel = GaussianKernel(data.train)

    ksvm = KSVM(kernel)
    ksvm.fit()

    data.show_class(ksvm.predict)
