from src.kernels.kernel import Kernel


class LinearKernel(Kernel):
    def __init__(self, dataset=None, verbose=True):
        super(LinearKernel, self).__init__(
            dataset=dataset, name="linear", verbose=verbose)

    def kernel(self, x, y):
        return x.dot(y)


if __name__ == "__main__":
    from src.tools.dataloader import GenClassData
    data = GenClassData(300, 3, nclasses=2, mode="gauss")
    kernel = LinearKernel(data.train)
    K = kernel.K
