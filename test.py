import argparse
from src.methods.KMethod import AllClassMethods
from src.data.dataset import AllClassData
from src.kernels.kernel import AllKernels
from src.methods.kpca import KPCA

methods, methodsNames = AllClassMethods()
datas, datasNames = AllClassData()
kernels, kernelsNames = AllKernels()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=methodsNames)
    parser.add_argument("--kernel", choices=kernelsNames)
    parser.add_argument("--data", choices=datasNames)
    parser.add_argument("--dim", type=int, choices=[2, 3], default=3)
    parser.add_argument("--show", type=bool, choices=[False, True], default=True)
    return parser.parse_args()


def find(name, allS):
    return allS.index(name)


if __name__ == "__main__":
    args = parse_args()
    print("Method: " + args.method)
    print("Kernel: " + args.kernel)
    print("Data: " + args.data)

    parameters = {"k": 2, "m": 1}
    # default parameter everywhere
    data = datas[find(args.data, datasNames)]()
    kernel = kernels[find(args.kernel, kernelsNames)](
        data, parameters=parameters)
    method = methods[find(args.method, methodsNames)](kernel)
    method.fit()

    arr = method.predict_array(data.data, binaire=False)
    
    # show
    if args.show:
        dim = args.dim
        kpca = KPCA(kernel, parameters={"dim": dim})
        proj = kpca.project()
        data.show_pca(proj, method.predictBin, dim=dim)
