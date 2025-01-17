import numpy as np
import heapq as hq
from src.methods.KMethod import KMethod, KMethodCreate, klogger


class KKNN(KMethod, metaclass=KMethodCreate):
    name = "kknn"
    defaultParameters = {"knn": 3}

    @klogger("Kernel k nearest neighbors")
    def fit(self, K):
        return None

    def predict(self, x):
        return self.majority_vote(x)

    def majority_vote(self, x):
        nn = self.nearest_neighbors(x)
        # Count the numbers of 1
        nb1 = np.sum(self.labels[nn] == 1)
        nb0 = self.param.knn - nb1
        if nb1 > nb0:
            return 1
        elif nb1 < nb0:
            return -1
        # Take uniform if it is not clear
        else:
            return np.random.choice([-1., 1.])

    def nearest_neighbors(self, x):
        K = self.kernel.K
        preds = self.kernel.predict(x)
        kxx = self.kernel(x, x)

        distance = []
        for i in range(self.n):
            dist = K[i, i] - 2 * preds[i] + kxx
            hq.heappush(distance, (dist, i))

        neighrest_neighboors = [
            hq.heappop(distance)[1] for _ in range(self.param.knn)
        ]
        return neighrest_neighboors


if __name__ == "__main__":
    from src.tools.test import EasyTest
    dparams = {"small": True, "nsmall": 300}
    EasyTest(kernels="spectral", data="seq", methods="kknn", dparams=dparams)
