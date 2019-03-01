import numpy as np
import heapq as hq
from src.tools.utils import sigmoid
from src.methods.KMethod import KMethod


class KKNN(KMethod):
    def __init__(self, kernel, lam=1, n_iter=100, verbose=True):
        super(KKNN, self).__init__(kernel=kernel, name="KKNN", verbose=verbose)
        self.lam = lam
        self.n_iter = n_iter

    def fit(self, dataset=None, labels=None, tol=10**-5, eps=10**-5):
        # Load the dataset (if there are one) in the kernel
        self.load_dataset(dataset, labels)

    def predict(self, x, j):
        return self.majority_vote(x, j)
    
    def majority_vote(self, x, j):
        nn = self.nearest_neighbors(x, j)
        N = len(self.labels)
        # Count the numbers of 1
        nb1 = np.sum(self.labels[nn] == 1)
        nb0 = N - nb1
        if nb1 > nb0:
            return 1
        elif nb1 < nb0:
            return -1
        # Take uniform if it is not clear
        else:
            return np.random.choice([-1., 1.])

    def nearest_neighbors(self, x, j):
        K = self.kernel.K
        preds = self.kernel.predict(x)
        kxx = self.kernel(x, x)

        distance = []
        for i in range(self.n):
            dist = K[i, i] - 2 * preds[i] + kxx
            hq.heappush(distance, (dist, i))

        neighrest_neighboors = [hq.heappop(distance)[1] for _ in range(j)]
        return neighrest_neighboors


if __name__ == "__main__":
    from src.tools.dataloader import SeqData
    data = SeqData(small=True)

    from src.kernels.spectral import SpectralKernel
    kernel = SpectralKernel(data.train)
    kknn = KKNN(kernel)
    kknn.fit()
    
    # data.show_class(klr.predict)
