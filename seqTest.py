import argparse
from src.methods.KMethod import AllClassMethods
from src.data.seq import AllSeqData
from src.kernels.kernel import AllStringKernels
from src.config import expPath
from src.tools.utils import submit, create_dir
from os import path

methods, methodsNames = AllClassMethods()
kernels, kernelsNames = AllStringKernels()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", choices=kernelsNames)
    parser.add_argument("method", choices=methodsNames)
    parser.add_argument(
        "--kparam", type=str, help="parameter for the kernel", default="")
    parser.add_argument(
        "--mparam", type=str, help="parameter for the method", default="")
    parser.add_argument("--csvname", default=None)
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


if __name__ == "__main__":
    args = parse_args()

    Kernel = kernels[find(args.kernel, kernelsNames)]
    kparams = parse_params(args.kparam, Kernel.defaultParameters)

    Method = methods[find(args.method, methodsNames)]
    mparams = parse_params(args.mparam, Method.defaultParameters)

    print("Kernel: " + args.kernel + ", with args: " + str(kparams))
    print("Method: " + args.method + ", with args: " + str(mparams))
    
    if args.csvname is None:
        create_dir(expPath)
        csvname = path.join(
            expPath,
            "kernel_" + str(kparams) + "_method_" + str(mparams) + ".csv")
    else:
        csvname = args.csvname
        
    print("Results will be saved in: " + csvname)
    
    datasets = AllSeqData()
    predictions = []
    Ids = []
    for dataset in datasets:
        # Fit on the train dataset
        train = dataset["train"]
        kernel = Kernel(train, parameters=kparams)
        method = Method(kernel, parameters=mparams)
        method.fit()

        # Compute the score of the train set:
        print(method.score_recall_precision(train))
        
        # Predictict on the test set and save the result
        test = dataset["test"]
        test.labels = method.predict_array(test.data)
        test.transform_label()
        predictions.append(test.labels)

        Ids.append(test.Id)

    submit(predictions, Ids, csvname)
