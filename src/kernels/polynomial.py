from src.kernels.kernel import DataKernel, KernelCreate


class PolynomialKernel(DataKernel, metaclass=KernelCreate):
    defaultParameters = {"offset": 0, "dim": 4}

    def _kernel(self, x, y):
        return (self.param.offset + x.dot(y))**self.param.dim


if __name__ == "__main__":
    from src.data.synthetic import GenClassData
    data = GenClassData(300, 3, nclasses=2)
    kernel = PolynomialKernel(data)

    from src.methods.kpca import KPCA
    kpca = KPCA(kernel)
    proj = kpca.project()

    data.show_pca(proj)
