from cvxopt import matrix, solvers
import numpy as np
from src.methods.KMethod import KMethod, KMethodCreate

solvers.options['show_progress'] = False

# Solve the QP Problem:
#  minimize    1/2 x^T*P*x + q^T*x
#  subject to  G*x <= h
#              A*x <= b
# solvers.qp(P, q, G, h)


class KSVM(KMethod, metaclass=KMethodCreate):
    """Documentation for ClassName
    Solve the KSVM problem:
        minimize    (1/2)*x^T*K*x - y^T*x
        subject to  0 <= yi*xi <= C
    """
    defaultParameters = {"C": 10**10}

    def fit(self, dataset=None, labels=None, K=None):
        # Load the dataset (if there are one) in the kernel
        self.load_dataset(dataset, labels)
        self._log("Fitting kernel svm..")
        
        n = self.n
        y = self.labels
        
        if K is None:
            K = self.kernel.K

        C = self.param.C

        P = matrix(K, (n, n), "d")
        q = matrix(-y, (n, 1), "d")
        G = matrix(np.concatenate((np.diag(y), np.diag(-y))), (2 * n, n), "d")
        h = matrix(
            np.concatenate((C * np.ones(n), np.zeros(n))), (2 * n, 1), "d")
        self._alpha = np.array(solvers.qp(P, q, G, h)["x"]).reshape(-1)

        self._log("Fitting done!")
        return self._alpha


if __name__ == "__main__":
    from src.data.synthetic import GenClassData
    data = GenClassData(500, 2, mode="circle")

    from src.kernels.quad import QuadKernel
    kernel = QuadKernel(data)
    
    ksvm = KSVM(kernel)
    ksvm.fit()

    f = (lambda x: ksvm.predict(x)*2 - 1)
    data._show_gen_class_predicted(f)
