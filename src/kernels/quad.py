from src.kernels.kernel import DataKernel, KernelCreate


class QuadKernel(DataKernel, metaclass=KernelCreate):
    defaultParameters = {"offset": 0}

    def kernel(self, x, y):
        return (x.dot(y) + self.param.offset)**2


if __name__ == "__main__":
    from src.data.synthetic import GenClassData
    data = GenClassData(300, 3, nclasses=2)
    kernel = QuadKernel(data)

    from src.methods.kpca import KPCA
    kpca = KPCA(kernel)
    proj = kpca.project()

    data.show_pca(proj)
