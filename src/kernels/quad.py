from src.kernels.kernel import Kernel


class QuadKernel(Kernel):
    def __init__(self, dataset=None, verbose=True):
        super(QuadKernel, self).__init__(
            dataset=dataset, name="quad", verbose=verbose)

    def kernel(self, x, y):
        return (x.dot(y) + 1)**2


if __name__ == "__main__":
    from src.tools.dataloader import GenClassData
    data = GenClassData(300, 3, nclasses=2, mode="gauss")
    kernel = QuadKernel(data.train, sigma=1, verbose=True)
    K = kernel.K
