from src.kernels.kernel import Kernel, EasyCreate


class LinearKernel(Kernel, metaclass=EasyCreate):
    defaultParameters = {"offset": 0}
    
    def kernel(self, x, y):
        return x.dot(y) + self.param.offset


if __name__ == "__main__":
    from src.data.synthetic import GenClassData
    data = GenClassData(300, 3, nclasses=2, mode="gauss")
    kernel = LinearKernel(data)
    K = kernel.K
