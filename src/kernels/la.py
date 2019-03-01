import numpy as np
from src.kernels.kernel import StringKernel, EasyCreate


class LAKernel(StringKernel, metaclass=EasyCreate):
    defaultParameters = {"e": 11, "d": 1, "beta": 0.5}
    S = np.array([[4, 0, 0, 0], [0, 9, -3, -1], [0, -3, 6, 2], [0, -1, -2, 5]])

    def affine_align(self, x, y):
        e = self.e
        d = self.d
        beta = self.beta
        S = self.S
        x, y = format(x) - 1, format(y) - 1
        n_x, n_y = len(x), len(y)
        M, X, Y, X2, Y2 = [np.zeros((n_x + 1, n_y + 1))] * 5
        for i in range(1, n_x):
            for j in range(1, n_y):
                M[i, j] = np.exp(beta * S[x[i], y[j]]) * (
                    1 + X[i - 1, j - 1] + Y[i - 1, j - 1] + M[i - 1, j - 1])
                X[i, j] = np.exp(beta * d) * M[i - 1, j] + np.exp(
                    beta * e) * X[i - 1, j]
                Y[i, j] = np.exp(beta * d) * (
                    M[i, j - 1] + X[i, j - 1]) + np.exp(beta * e) * Y[i, j - 1]
                X2[i, j] = M[i - 1, j] + X2[i - 1, j]
                Y2[i, j] = M[i, j - 1] + X2[i, j - 1] + Y2[i, j - 1]
        return (
            1 / beta) * np.log(1 + X2[n_x, n_y] + Y2[n_x, n_y] + M[n_x, n_y])

    def Smith_Waterman(self, x, y, e=11, d=1, beta=0.5):
        S = self.S
        x, y = format(x) - 1, format(y) - 1
        n_x, n_y = len(x), len(y)
        M, X, Y, X2, Y2 = [np.zeros((n_x + 1, n_y + 1))] * 5
        for i in range(1, n_x):
            for j in range(1, n_y):
                M[i, j] = np.exp(beta * S[x[i], y[j]]) * max(
                    1, X[i - 1, j - 1], Y[i - 1, j - 1], M[i - 1, j - 1])
                X[i, j] = max(
                    np.exp(beta * d) * M[i - 1, j],
                    np.exp(beta * e) * X[i - 1, j])
                Y[i, j] = max(
                    np.exp(beta * d) * M[i, j - 1],
                    np.exp(beta * d) * X[i, j - 1],
                    np.exp(beta * e) * Y[i, j - 1])
                X2[i, j] = max(M[i - 1, j], X2[i - 1, j])
                Y2[i, j] = max(M[i, j - 1], X2[i, j - 1], Y2[i, j - 1])
        return (1 / beta) * np.log(
            max(1, X2[n_x, n_y], Y2[n_x, n_y], M[n_x, n_y]))

    def get_LA_K(self, X, e=11, d=1, beta=0.5, smith=0, eig=1):
        n = X.shape[0]
        K = np.zeros((n, n))
        for i, x in tqdm(
                enumerate(X.loc[:, 'seq']), total=n, desc='Building kernel'):
            for j, y in tqdm(enumerate(X.loc[:, 'seq']), total=n - i):
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


if __name__ == "__main__":
    from src.data.seq import SeqData
    data = SeqData(small=True)
    kernel = LAKernel(data)
    K = kernel.K
