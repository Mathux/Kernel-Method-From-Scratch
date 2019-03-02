from src.kernels.kernel import Kernel, KernelCreate


class QuadKernel(Kernel, metaclass=KernelCreate):
    defaultParameters = {"offset": 0}

    def kernel(self, x, y):
        return (x.dot(y) + self.param.offset)**2


if __name__ == "__main__":
    from src.data.synthetic import GenClassData
    data = GenClassData(300, 3, nclasses=2, mode="gauss")
    kernel = QuadKernel(data, verbose=True)
    K = kernel.K
