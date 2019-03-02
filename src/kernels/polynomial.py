from src.kernels.kernel import Kernel, KernelCreate


class PolynomialKernel(Kernel, metaclass=KernelCreate):
    defaultParameters = {"offset": 0, "dim": 4}

    def kernel(self, x, y):
        return (self.param.offset + x.dot(y))**self.param.dim


if __name__ == "__main__":
    from src.data.synthetic import GenClassData
    data = GenClassData(300, 3, nclasses=2)
    kernel = PolynomialKernel(data)

    from src.methods.kpca import KPCA
    kpca = KPCA(kernel)
    proj = kpca.project()

    data.show_pca(proj)
