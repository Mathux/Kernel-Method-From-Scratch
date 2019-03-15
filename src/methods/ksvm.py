from cvxopt import matrix, solvers
import numpy as np
from src.methods.KMethod import KMethod, KMethodCreate, klogger

solvers.options['show_progress'] = False

# Solve the QP Problem:
#  minimize    1/2 x^T*P*x + q^T*x
#  subject to  G*x <= h
#              A*x <= b
# solvers.qp(P, q, G, h)


class KSVM(KMethod, metaclass=KMethodCreate):
    """Documentation for KSVM
    Solve the KSVM problem:
        minimize    (1/2)*x^T*K*x - y^T*x
        subject to  0 <= yi*xi <= C
    """
    name = "ksvm"
    defaultParameters = {"C": 1.0, "tol": 10**-4}
    
    @klogger("Kernel Support Vector Machine")
    def fit(self, K):
        n = self.n
        y = self.labels
        C = self.param.C

        P = matrix(K, (n, n), "d")
        q = matrix(-y, (n, 1), "d")
        G = matrix(np.concatenate((np.diag(y), np.diag(-y))), (2 * n, n), "d")
        h = matrix(
            np.concatenate((C * np.ones(n), np.zeros(n))), (2 * n, 1), "d")
        alpha = np.array(solvers.qp(P, q, G, h)["x"]).reshape(-1)

        # support_vectors = np.where(np.abs(alpha) > self.param.tol)

        self._alpha = alpha
        return self._alpha


if __name__ == "__main__":
    from src.tools.test import EasyTest
    dparams = {"small": False, "nsmall": 300}
    EasyTest(kernel="spectral", data="seq", method="ksvm", dparams=dparams)
    
    # from src.kernels.wildcard_trie import WildcardTrieKernel
    # kernel = WildcardTrieKernel(data)
