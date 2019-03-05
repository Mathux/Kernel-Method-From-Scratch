import numpy as np
from src.kernels.kernel import DataKernel, KernelCreate


# The local alignement kernel
class LAKernel(DataKernel, metaclass=KernelCreate):
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
        for i in range(1, nx+1):
            for j in range(1, ny+1):
                c = (i - 1, j - 1)
                M[i, j] = np.exp(beta * S(x[i-1], y[j-1]))
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
    data = SeqData(small=True, nsmall=50)
        
    kernel = LAKernel(data)

    from src.methods.kpca import KPCA
    kpca = KPCA(kernel)
    proj = kpca.project()

    data.show_pca(proj)
