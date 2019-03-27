import argparse
from src.methods.KMethod import AllClassMethods
from src.data.dataset import AllClassData
from src.kernels.kernel import AllKernels
from src.methods.kpca import KPCA
from src.config import expPath
from src.tools.utils import submit, create_dir, Logger, objdict
from os import path
import sys
import json

# import ipdb

methods, methodsNames = AllClassMethods()
datas, datasNames = AllClassData()
kernels, kernelsNames = AllKernels()


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help(sys.stderr)
        self.exit(2, '%s: error: %s\n' % (self.prog, message))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("kernels", choices=kernelsNames, help="kernel name")
    parser.add_argument("methods", choices=methodsNames, help="method name")
    parser.add_argument("data", choices=datasNames, help="data name")
    parser.add_argument("--kparams", default=None, help="kernel parameters")
    parser.add_argument("--mparams", default=None, help="method parameters")
    parser.add_argument("--dparams", default=None, help="data parameters")
    parser.add_argument(
        "--pcadim",
        default=3,
        choices=[2, 3],
        type=int,
        help="pca dimention for visualization")
    parser.add_argument(
        "--show", type=bool, default=False, help="True to draw the pca")
    parser.add_argument(
        "--csvname", default=None, help="Path to the csv file to write")
    parser.add_argument(
        "--submit", type=bool, default=False, help="Submit a csv or not")
    return parser.parse_args()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_params(allargs, defaultParams):
    if allargs is None:
        return defaultParams

    parser = ArgumentParser()

    for args, default in defaultParams.items():
        if type(default) == bool:
            parser.add_argument(
                "--" + args,
                type=str2bool,
                nargs='?',
                const=True,
                default=default,
                help="Activate " + args + " mode.")
        else:
            parser.add_argument(
                "--" + args, default=default, type=type(default))

    return parser.parse_args(allargs.split(" ")).__dict__


def find(name, allS):
    return allS.index(name)


def findKernel(name):
    return kernels[find(name, kernelsNames)]


def findMethod(name):
    if name is None:
        return NoneMethod
    return methods[find(name, methodsNames)]


def findData(name):
    return datas[find(name, datasNames)]


class NoneMethod:
    """Dummy method for concision

    """

    def __init__(self, kernel, parameters):
        self.predictBin = None
        return

    def fit(self):
        return None

    def score_recall_precision(self, dataset, nsmall=None):
        return None

    def sanity_check(self):
        return None

    def __str__(self):
        name = "No Method tested here"
        return name


def find_more_or_one(string_or_list, find_func, n):
    if isinstance(string_or_list, list):
        return [find_func(x) for x in string_or_list]
    else:
        return [find_func(string_or_list) for _ in range(n)]


def KernelTest(kernelname, parameters, synth=False):
    Dataset = findData("allseq")()[0]
    if synth:
        import numpy as np
        from src.data.dataset import Dataset

        defaultParameters = {
            "k": 0,
            "mat": False,
            "shuffle": False,
            "small": False,
            "nsmall": 200,
            "labels_change": True,
            "name": "seq",
            "nclasses": 2
        }

        from src.tools.utils import Parameters
        p = Parameters(None, defaultParameters)

        train = Dataset(p, np.array(['ATTA', 'AAAA']), np.array([0, 1]))
    else:
        train = Dataset["train"]
    
    Kernel = findKernel(kernelname)

    Logger.log(True, "Test the " + kernelname + " kernel.")
    Logger.indent()
    kernels = []
    for params in parameters:
        Logger.log(True, "Test with these parameters: " + str(params))
        Logger.indent()
        kernel = Kernel(train, params)
        kernels.append(kernel)
        Logger.log(True, kernel.K)
        Logger.dindent()

    # ipdb.set_trace()
    Logger.dindent()
    
    
def EasyTest(kernels,
             data="seq",
             methods=None,
             dparams=None,
             kparams=None,
             mparams=None,
             pcadim=3,
             show=False,
             dopredictions=False,
             verbose=True):

    Datasets = findData(data)(dparams, verbose)
    ndatas = len(Datasets)

    Kernels = find_more_or_one(kernels, findKernel, ndatas)
    KMethods = find_more_or_one(methods, findMethod, ndatas)
    Kparams = find_more_or_one(kparams, lambda x: x, ndatas)
    Mparams = find_more_or_one(mparams, lambda x: x, ndatas)

    predictions = []
    Ids = []
    scores = []

    Logger.indent()
    for Dataset, Kernel, KMethod, Kparam, Mparam in zip(
            Datasets, Kernels, KMethods, Kparams, Mparams):
        Logger.dindent()
        Logger.log(verbose, "Experiment on:")
        Logger.indent()

        train = Dataset["train"]
        # train._show_gen_class_data()
        kernel = Kernel(train, parameters=Kparam)
        method = KMethod(kernel, parameters=Mparam)

        Logger.log(verbose, kernel)
        Logger.log(verbose, method)
        Logger.log(verbose, train)
        Logger.log(verbose, "")

        method.fit()

        # Logger.log(verbose, method.alpha)
        # Check the value to see if it is alright
        method.sanity_check()
        # Compute the score of the train set:
        score = method.score_recall_precision(train, nsmall=200)
        scores.append(score)
        if show:
            Logger.log(verbose, "Show the trainset in the feature space..")
            Logger.indent()

            kpca = KPCA(kernel, parameters={"dim": pcadim})
            proj = kpca.project()
            predict = method.predict_array(train.data, desc="Projections")

            Logger.dindent()
            kernel.dataset.show_pca(proj, predict, dim=pcadim)

        if dopredictions:
            # Predictict on the test set and save the result
            test = Dataset["test"]
            test.labels = method.predict_array(test.data)
            test.transform_label()
            predictions.append(test.labels)

            Ids.append(test.Id)

    Logger.dindent()

    Logger.log(verbose, "Score remainder:")
    Logger.indent()
    [Logger.log(verbose, s) for s in scores]
    Logger.dindent()

    if dopredictions:
        return scores, predictions, Ids

    else:
        return scores


if __name__ == "__main__":
    # If we call this function with the link testmodel
    if "testmodel" in sys.argv[0]:
        args = objdict(json.load(open("config.json", "r")))
        args.csvname = None
        kparams = args.kparams
        mparams = args.mparams
        dparams = args.dparams
    else:
        args = parse_args()
        kparams = parse_params(args.kparams,
                               findKernel(args.kernels).defaultParameters)
        mparams = parse_params(args.mparams,
                               findMethod(args.methods).defaultParameters)
        dparams = parse_params(args.dparams,
                               findData(args.data).defaultParameters)

    if args.csvname is None:
        create_dir(expPath)
        # sd = args.data + "_data_"
        # sk = args.kernels + "_kernel(" + str(kparams) + ")_"
        # sm = args.methods + "_method(" + str(mparams) + ")"
        # TODO, find a way to store all the infos
        name = "experiment"
        csvname = path.join(expPath, name + ".csv")
    else:
        csvname = args.csvname

    if args.submit:
        print("Results will be saved in: " + csvname)
        print()

    scores = EasyTest(
        kernels=args.kernels,
        data=args.data,
        methods=args.methods,
        dparams=dparams,
        kparams=kparams,
        mparams=mparams,
        pcadim=args.pcadim,
        show=args.show,
        dopredictions=args.submit,
        verbose=True)

    if args.submit:
        scores, predictions, Ids = scores
        submit(predictions, Ids, csvname)

        print("Results saved in: " + csvname)
