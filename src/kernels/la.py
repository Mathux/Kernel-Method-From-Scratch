import numpy as np
from src.kernels.kernel import Kernel, KernelCreate


# The local alignement kernel
class LAKernel(Kernel, metaclass=KernelCreate):
    S = np.array([[4, 0, 0, 0], [0, 9, -3, -1], [0, -3, 6, 2], [0, -1, -2, 5]])
    defaultParameters = {"e": 11, "d": 1, "beta": 0.5, "S": S, "mode": "smith"}

    def kernel(self, x, y):
        if self.param.mode == "smith":
            op = max
        elif self.param.mode == "affine_align":
            op = sum

        e = self.param.e
        d = self.param.d
        beta = self.param.beta

        def S(x, y):
            return self.param.S[x, y]

        x, y = LAKernel.format(x), LAKernel.format(y)
        nx, ny = len(x), len(y)
        M, X, Y, X2, Y2 = [np.zeros((nx + 1, ny + 1)) for _ in range(5)]
        for i in range(1, nx):
            for j in range(1, ny):
                c = (i - 1, j - 1)
                M[i, j] = np.exp(beta * S(x[i], y[j]))
                M[i, j] *= op([1, X[c], Y[c], M[c]])

                BD = np.exp(beta * d)
                BE = np.exp(beta * e)
                
                c = (i - 1, j)                
                X[i, j] = op([BD * M[c], BE * X[c]])
                X2[i, j] = op([M[c], X2[c]])
                
                c = (i, j - 1)
                Y[i, j] = op([BD * M[c], BD * X[c], BE * Y[c]])
                Y2[i, j] = op([M[c], X2[c], Y2[c]])
                
        return (1 / beta) * np.log(op([1, X2[nx, ny], Y2[nx, ny], M[nx, ny]]))

    @staticmethod
    def format(x):
        """
        Transform string 'AGCT' to list [0, 2, 1, 3]
        :param x: string, DNA sequence
        :return: np.array, array of ints with 'A':0, 'C':1, 'G':2, 'T':3
        """
        return np.array(list(x.replace('A', '0').replace('C', '1').replace('G', '2').replace('T', '3')), dtype=np.int64)


if __name__ == "__main__":
    from src.data.seq import SeqData
    data = SeqData(small=True)
    kernel = LAKernel(data, parameters={"mode": "smith"})
    K = kernel.K


DISCUSS = """
    def get_LA_K(self, X, e=11, d=1, beta=0.5, smith=0, eig=1):
        n = X.shape[0]
        K = np.zeros((n, n))
        for i, x in self.viterator(
                enumerate(X.loc[:, 'seq']), desc='Building kernel'):
            for j, y in enumerate(X.loc[:, 'seq']):
                if j >= i:
                    K[i, j] = self.Smith_Waterman(
                        x, y, e, d, beta) if smith else self.affine_align(
                            x, y)
                    K[j, i] = K[i, j]
        K1 = deepcopy(K)
        if eig == 1:
            vp = np.min(np.real(eigs(K1)[0]))
            s = vp if vp < 0 else 0
            np.fill_diagonal(K1, np.diag(K1) - s * np.ones(n))
        else:
            for i in tqdm(range(K1.shape[0]), desc='Empirical kernel'):
                for j in range(i, n):
                    K1[i, j] = np.dot(K[i], K[j])
                    K1[j, i] = K1[i, j]
        return K
"""
