from src.kernels.kernel import DataKernel, KernelCreate


class LinearKernel(DataKernel, metaclass=KernelCreate):
    defaultParameters = {"offset": 0}

    def _kernel(self, x, y):
        return x.dot(y) + self.param.offset


if __name__ == "__main__":
    from src.data.synthetic import GenClassData
    data = GenClassData(300, 3, nclasses=2)
    kernel = LinearKernel(data)

    from src.methods.kpca import KPCA
    kpca = KPCA(kernel)
    proj = kpca.project()

    data.show_pca(proj)
