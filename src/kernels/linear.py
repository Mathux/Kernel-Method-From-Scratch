from src.kernels.kernel import DataKernel, KernelCreate


class LinearKernel(DataKernel, metaclass=KernelCreate):
    name = "linear"
    defaultParameters = {"offset": 0}

    def kernel(self, x, y):
        return x.dot(y) + self.param.offset


if __name__ == "__main__":
    from src.tools.test import EasyTest
    EasyTest(kernel="linear", data="synth")

    #import src.data.kernelLoader as kernelLoader
    #pathKernel = kernelLoader.save(kernel)

    #test = kernelLoader.load(pathKernel)
