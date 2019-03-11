# Kernel Method Project
Author: Mathis Petrovich, Evrard Garcelon


## Make a submission

String kernels available:
- mismatch
- spectral
- wd
- la

Methods availables:
- kknn
- klr
- ksvm

Test a kernel method with a kernel:

```bash
python -i seqTest.py spectral ksvm --kparam "--k 7" --mparam "--C 1 --tol 0.01"
```
