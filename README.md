# Quasi-Newton optimization algorithms to enhance PINNs

This repository includes the implementation of the Self-scaled Quasi-Newton algorithms suggested in:
* [Unveiling the optimization process of Physics Informed Neural Networks: How accurate and competitive can PINNs be?](https://www.sciencedirect.com/science/article/pii/S0021999124009045)

together with numerous scripts corresponding to the examples discussed in this work. These scripts contain numerous comments, in order to facilitate their use and modification. Other optimization algorithms will also be included in the future.

# Requirements
* The Machine learning frameworks considered here are [Tensorflow](https://www.tensorflow.org/?hl=es-419) y [Keras](https://keras.io/). The specific versions used in these work are 2.10.1 for Tensorflow, and 2.10.0 for Keras.
* [Numpy](https://numpy.org/) and [Scipy](https://scipy.org/) are also needed. This work has used 1.24.1 for Numpy, and 1.12.0 for Scipy.

# How to use

* In order to use the codes associated with each of the examples, you should first download the codes `modified_minimize.py` and `modified_optimize.py`, and then replace respectively the scripts `_minimize.py` and `_optimize.py` by these two files. You will find the scripts `_minimize.py` and `_optimize.py` in a folder called `optimize`, where you have Scipy installed.
* Each folder contains of this repository within the folder `examples` contains two `.py` files: 
  - Main file: Same name as the folder. This is the file used for training.
  - Hyperparameter file: Same name as the main file, but followed by `_hparams`. In this file, you can choose:
    + Architecture hyperparameters: Hidden layers, neurons at every hidden layer, and output neurons.
    + PDE parameters (if any)
    + Adam hyperparameters: All the hyperparameters related with Adam optimization. See the Main file for details.
    + Batch hyperparameters: Number of points, number of iterations per batch, and adaptive resampling hyperparameters. We have incorporated here the RAD algorithm introduced in [A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks](https://www.sciencedirect.com/science/article/abs/pii/S0045782522006260).
    + Quasi-Newton hyperparameters: All the hyperparameters related with Quasi-Newton optimization. See the Main file for details.
      
* Within the different options for the Quasi-Newton hyperparameters, we can select the different Self-scaled Quasi-Newton algorithms, as well as other additional algorithms. In order to choose them we have two different variables: `method` and `method_bfgs`:
   - In `method`, you can choose between:
     + `BFGS`: Here, we include BFGS and the different self-scaled QN methods. To distinguish between the different QN algorithms, we use `method_bfgs` (see below).
     + `bfgsr`: Personal implementation of the factored BFGS Hessian approximations. See [On Recent Developments in BFGS Methods for Unconstrained Optimization](https://ccom.ucsd.edu/reports/UCSD-CCoM-22-01.pdf) for details.
     + `bfgsz`: Personal implementation of the factored inverse BFGS Hessian approximations. See [On Recent Developments in BFGS Methods for Unconstrained Optimization](https://ccom.ucsd.edu/reports/UCSD-CCoM-22-01.pdf) for details.
   - If `method=BFGS`, the variable `method_bfgs` chooses different QN methods. We have implemented:
     + `BFGS_scipy`: The original implementation of BFGS of Scipy.
     + `BFGS`: Equivalent implementation, but faster.
     + `SSBFGS_AB`: The Self-scaled BFGS formula, where the tauk coefficient is introduced originally in [Numerical Experience with a Class of Self-Scaling Quasi-Newton Algorithms](https://link.springer.com/article/10.1023/A:1022608410710) (see also expression 11 of [our article](https://www.sciencedirect.com/science/article/pii/S0021999124009045)).
     + `SSBFGS_OL` Same, but tauk is calculated with the original choice of [Self-Scaling Variable Metric (SSVM) Algorithms](https://pubsonline.informs.org/doi/10.1287/mnsc.20.5.845).
     + `SSBroyden2`: Here we use the tauk and phik expressions defined in the paper
       (Formulas 13-23 of [our article](https://www.sciencedirect.com/science/article/pii/S0021999124009045))
     + `SSBroyden1`: Another possible choice for these parameters (sometimes better, sometimes worse than `SSBroyden2`).
*Finally, you can also choose to train against the square root of the logarithm of the Loss function. To do that, choose `use_sqrt = True` or `use_log = True` respectively in the Hyperparameter file.
  
# IMPORTANT
This repository contains modified versions of two scripts from the ‘optimize’ package within the Scipy library [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html). Specifically, the scripts named ‘_optimize.py’ and ‘_minimize.py’ have been modified. The original codes can be found in [scipy](https://github.com/scipy/scipy/tree/main/scipy/optimize). 

This project is NOT officially maintained or endorsed by the original developers of SciPy. The use of the name 'SciPy' or any related names should not be interpreted as endorsement, sponsorship, or support by the SciPy developers.

# License
This project uses the BSD 3-clause license. This work uses modified scripts of Scipy (see `LICENSE.txt`).

# Citation 
```bibtex
@article{UrbanStefanouPons2024,
title = {Unveiling the optimization process of Physics Informed Neural Networks: How accurate and competitive can PINNs be?},
journal = {Journal of Computational Physics},
pages = {113656},
year = {2024},
issn = {0021-9991},
doi = {https://doi.org/10.1016/j.jcp.2024.113656},
url = {https://www.sciencedirect.com/science/article/pii/S0021999124009045},
author = {Jorge F. Urbán and Petros Stefanou and José A. Pons}
}
