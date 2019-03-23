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
        A = matrix(np.ones(n), (1, n), "d")

        b = matrix(0.0) 
        alpha = np.array(solvers.qp(P, q, G, h, A = A, b = b)["x"]).reshape(-1)

        support_vectors = np.where(np.abs(alpha) > self.param.tol)[0]
        intercept = 0
        for sv in support_vectors:
            intercept += y[sv]
            intercept -= np.sum(alpha[support_vectors] * K[sv, support_vectors])
        intercept /= len(support_vectors)
        
        self._b = intercept
        self._alpha = alpha
        return self._alpha, self._b


if __name__ == "__main__":
    from src.tools.test import EasyTest
    dparams = {"small": True, "nsmall": 200}
    EasyTest(kernels="spectral", data="seq", methods="ksvm", show = True, dparams=dparams)
    
#     from src.kernels.wildcard_trie import WildcardTrieKernel
#     kernel = WildcardTrieKernel(data)
