# Kernel Method Project
Author: Mathis Petrovich, Evrard Garcelon

Project from the MVA course Kernel methods for machine learning.

## Introduction 
This [Kaggle](https://www.kaggle.com/c/kernel-methods-for-machine-learning-2018-2019 "Kaggle Kernel Methods") competition is focused on the prediction of
binding sites to a certain transcription factor for DNA sequences using kernel methods. To answer this question, we
implemented several kernel classifiers and several kernels
designed for protein sequences.


## Leaderboard
The model we choose for the final submission was the sum of spectrum kernel for k = 7 to 20. This final submission scored 0.71333 on the public leaderboard (ranked at the 21th place) and scored 0.71 on the private leaderboard (ranked 3rd place).


## Recreate this submission
Just run this command in a terminal:
```bash
./start
```

or

```bash
python start.py
```

## Test other kernels/methods
String kernels available:
- mismatch
- spectral
- spectralconcat (for adding spectral together)
- wd
- la
- wildcard
- etc

Methods availables:
- kknn
- klr
- ksvm

### Exemple 
```bash
python -m src.tools.test spectral ksvm allseq --kparam "--k 7" --mparam "--C 1 --tol 0.01" --submit True --cvsname "Yte.csv" --show True --pcadim 3
```
### General usage

```bash
usage: test.py [-h] [--kparams KPARAMS] [--mparams MPARAMS]
               [--dparams DPARAMS] [--pcadim {2,3}] [--show SHOW]
               [--csvname CSVNAME] [--submit SUBMIT]
               {mismatch,spectral,wd,la,wildcard,gappy,spectralconcat,exp,gaussian,laplacian,linear,poly,quad,sigmoid,multikernel}
               {kknn,klr,ksvm,simple_mkl} {allseq,seq,synth}

positional arguments:
  {mismatch,spectral,wd,la,wildcard,gappy,spectralconcat,exp,gaussian,laplacian,linear,poly,quad,sigmoid,multikernel}
                        kernel name
  {kknn,klr,ksvm,simple_mkl}
                        method name
  {allseq,seq,synth}    data name

optional arguments:
  -h, --help            show this help message and exit
  --kparams KPARAMS     kernel parameters
  --mparams MPARAMS     method parameters
  --dparams DPARAMS     data parameters
  --pcadim {2,3}        pca dimention for visualization
  --show SHOW           True to draw the pca
  --csvname CSVNAME     Path to the csv file to write
  --submit SUBMIT       Submit a csv or not
```