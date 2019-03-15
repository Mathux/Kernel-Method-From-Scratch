import argparse
from src.methods.KMethod import AllClassMethods
from src.data.dataset import AllClassData
from src.kernels.kernel import AllKernels
from src.methods.kpca import KPCA
from src.config import expPath
from src.tools.utils import submit, create_dir, Logger
from os import path
import sys

methods, methodsNames = AllClassMethods()
datas, datasNames = AllClassData()
kernels, kernelsNames = AllKernels()


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help(sys.stderr)
        self.exit(2, '%s: error: %s\n' % (self.prog, message))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("kernel", choices=kernelsNames, help="kernel name")
    parser.add_argument("method", choices=methodsNames, help="method name")
    parser.add_argument("data", choices=datasNames, help="data name")
    parser.add_argument("--kparams", default="", help="kernel parameters")
    parser.add_argument("--mparams", default="", help="method parameters")
    parser.add_argument("--dparams", default="", help="data parameters")
    parser.add_argument(
        "--pcadim",
        default=3,
        choices=[2, 3],
        type=int,
        help="pca dimention for visualization")
    parser.add_argument(
        "--show", type=bool, default=True, help="True to draw the pca")
    parser.add_argument(
        "--csvname", default=None, help="Path to the csv file to write")
    parser.add_argument(
        "--submit", type=bool, default=False, help="Submit a csv or not")
    return parser.parse_args()


def parse_params(allargs, defaultParams):
    parser = argparse.ArgumentParser()

    for args, default in defaultParams.items():
        parser.add_argument("--" + args, default=default, type=type(default))

    if allargs == "":
        return defaultParams
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

    def score_recall_precision(self, dataset):
        return None

    def sanity_check(self):
        return None
    
    def __str__(self):
        name = "No Method tested here"
        return name


def EasyTest(kernel,
             data="seq",
             method=None,
             dparams=None,
             kparams=None,
             mparams=None,
             pcadim=3,
             show=True,
             dopredictions=False,
             verbose=True):

    Data = findData(data)
    Kernel = findKernel(kernel)
    KMethod = findMethod(method)
    
    predictions = []
    Ids = []
    scores = []

    datasets = Data(dparams, verbose)    
    
    Logger.indent()
    for dataset in datasets:
        Logger.dindent()
        Logger.log(verbose, "Experiment on:")
        Logger.indent()

        train = dataset["train"]
        kernel = Kernel(train, parameters=kparams)
        method = KMethod(kernel, parameters=mparams)

        Logger.log(verbose, kernel)
        Logger.log(verbose, method)
        Logger.log(verbose, train)
        Logger.log(verbose, "")
        
        method.fit()

        # Check the value to see if it is alright
        method.sanity_check()
        
        # Compute the score of the train set:
        score = method.score_recall_precision(train)
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
            test = dataset["test"]
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
    args = parse_args()

    kparams = parse_params(args.kparams,
                           findKernel(args.kernel).defaultParameters)
    mparams = parse_params(args.mparams,
                           findMethod(args.method).defaultParameters)
    dparams = parse_params(args.dparams, findData(args.data).defaultParameters)
    
    if args.csvname is None:
        create_dir(expPath)
        sd = args.data + "_data_"
        sk = args.kernel + "_kernel(" + str(kparams) + ")_"
        sm = args.method + "_method(" + str(mparams) + ")"
        csvname = path.join(
            expPath,
            sd + sk + sm + ".csv")
    else:
        csvname = args.csvname
    
    if args.submit:
        print("Results will be saved in: " + csvname)
        print()
    
    scores = EasyTest(
        kernel=args.kernel,
        data=args.data,
        method=args.method,
        dparams=dparams,
        kparams=kparams,
        mparams=mparams,
        pcadim=args.pcadim,
        show=True,
        dopredictions=args.submit,
        verbose=True)

    if args.submit:
        scores, predictions, Ids = scores
        submit(predictions, Ids, csvname)
        
        print("Results saved in: " + csvname)

