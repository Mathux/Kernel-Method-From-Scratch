from src.kernels.kernel import DataKernel, KernelCreate


class QuadKernel(DataKernel, metaclass=KernelCreate):
    name = "quad"
    defaultParameters = {"offset": 10}
    
    def kernel(self, x, y):
        return (x.dot(y) + self.param.offset)**2


if __name__ == "__main__":
    from src.tools.test import EasyTest
    EasyTest(kernel="quad", data="synth")
