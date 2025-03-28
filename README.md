# Quasi-Newton optimization algorithms to enhance PINNs

This repository includes the implementation of the Self-scaled Quasi-Newton algorithms suggested in:
* [Unveiling the optimization process of Physics Informed Neural Networks: How accurate and competitive can PINNs be?](https://www.sciencedirect.com/science/article/pii/S0021999124009045)

together with numerous scripts corresponding to the examples discussed in this work. These scripts contain numerous comments, in order to facilitate their use and modification. Other optimization algorithms, as well as other examples, will also be included in the future.

# Requirements
* The Machine learning frameworks considered here are [Tensorflow](https://www.tensorflow.org/?hl=es-419) and [Keras](https://keras.io/). The specific versions used in this work are 2.10.1 for Tensorflow, and 2.10.0 for Keras.
* [Numpy](https://numpy.org/) and [Scipy](https://scipy.org/) are also needed. This work has used 1.24.1 for Numpy, and 1.12.0 for Scipy.

# How to use

* To use the codes associated with each example, please download the `modified_minimize.py` and `modified_optimize.py` files and replace the `_minimize.py` and `_optimize.py` scripts with these two files, respectively. The `_minimize.py` and `_optimize.py` scripts can be found in a folder called `optimize`, which is located  within the SciPy folder.
* The examples referenced in [our article](https://www.sciencedirect.com/science/article/pii/S0021999124009045) are within the folder `Examples` and saved in different subfolders. These folders have been named according to the problem they refer to.
* Each of these folders contains two `.py` files: 
  - **Main file**: Same name as the folder. This is the file used for training.
  - **Hyperparameter file**: Same name as the main file, but followed by `_hparams`. In this file, you can choose:
    + **Architecture hyperparameters**: Hidden layers, neurons at every hidden layer, and output neurons.
    + **PDE parameters (if any)**
    + **Adam hyperparameters**: All the hyperparameters related with Adam optimization. See the Main file for details.
    + **Batch hyperparameters**: Number of points, number of iterations per batch, and adaptive resampling hyperparameters. We have incorporated here the RAD algorithm introduced in [A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks](https://www.sciencedirect.com/science/article/abs/pii/S0045782522006260).
    + **Quasi-Newton hyperparameters**: All the hyperparameters related with Quasi-Newton optimization. See the **Main file** for details.
* Run first the **hyperparameter file**, to generate a `.json` file with all the hyperparameters. Next, run the **main file**.      
* Within the different options for the **Quasi-Newton hyperparameters**, we can select the different Self-scaled Quasi-Newton algorithms, as well as other additional algorithms. In order to choose between them we have two different variables: `method` and `method_bfgs`:
   - In `method`, you can choose between:
     + `BFGS`: Here, we include BFGS and the different self-scaled Quasi-Newton methods. To distinguish between the different Quasi-Newton algorithms, we use `method_bfgs` (see below).
     + `bfgsr`: Personal implementation of the factored BFGS Hessian approximations. See [On Recent Developments in BFGS Methods for Unconstrained Optimization](https://ccom.ucsd.edu/reports/UCSD-CCoM-22-01.pdf) for details.
     + `bfgsz`: Personal implementation of the factored inverse BFGS Hessian approximations. See [On Recent Developments in BFGS Methods for Unconstrained Optimization](https://ccom.ucsd.edu/reports/UCSD-CCoM-22-01.pdf) for details.
   - If `method=BFGS`, the variable `method_bfgs` chooses different Quasi-Newton methods. We have implemented:
     + `BFGS_scipy`: The original implementation of BFGS of Scipy.
     + `BFGS`: Equivalent implementation, but faster.
     + `SSBFGS_AB`: The Self-scaled BFGS formula, where the tauk coefficient is introduced originally in [Numerical Experience with a Class of Self-Scaling Quasi-Newton Algorithms](https://link.springer.com/article/10.1023/A:1022608410710) (see also expression 11 of [our article](https://www.sciencedirect.com/science/article/pii/S0021999124009045)).
     + `SSBFGS_OL` Same, but tauk is calculated with the original choice of [Self-Scaling Variable Metric (SSVM) Algorithms](https://pubsonline.informs.org/doi/10.1287/mnsc.20.5.845).
     + `SSBroyden2`: Here we use the tauk and phik expressions originally introduced in [A Wide Interval for Efficient Self-Scaling Quasi-Newton Algorithms](https://optimization-online.org/2003/08/699/)
       (Formulas 13-23 of [our article](https://www.sciencedirect.com/science/article/pii/S0021999124009045))
     + `SSBroyden1`: Another possible choice for these parameters  introduced in [A Wide Interval for Efficient Self-Scaling Quasi-Newton Algorithms](https://optimization-online.org/2003/08/699/) (sometimes better, sometimes worse than `SSBroyden2`).
* Finally, you can also choose to train against the square root or the logarithm of the loss function during the Quasi-Newton optimization. To do that, choose `use_sqrt = True` or `use_log = True` respectively in the Hyperparameter file.
  
# IMPORTANT
This repository contains modified versions of two scripts from the ‘optimize’ package within the Scipy library [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html). Specifically, the scripts named ‘_optimize.py’ and ‘_minimize.py’ have been modified. The original codes can be found in [scipy](https://github.com/scipy/scipy/tree/main/scipy/optimize). 

This project is NOT officially maintained or endorsed by the original developers of SciPy. The use of the name 'SciPy' or any related names should not be interpreted as endorsement, sponsorship, or support by the SciPy developers.

# License
This project uses the BSD 3-clause license. This work uses modified scripts of Scipy (see `LICENSE.txt`).

# Citation 
```bibtex
@article{UrbanStefanouPons2025,
title = {Unveiling the optimization process of physics informed neural networks: How accurate and competitive can PINNs be?},
journal = {Journal of Computational Physics},
volume = {523},
pages = {113656},
year = {2025},
issn = {0021-9991},
doi = {https://doi.org/10.1016/j.jcp.2024.113656},
url = {https://www.sciencedirect.com/science/article/pii/S0021999124009045},
author = {Jorge F. Urbán and Petros Stefanou and José A. Pons},
keywords = {Physics-informed neural networks, Optimization algorithms, Non-linear PDEs},
abstract = {This study investigates the potential accuracy boundaries of physics-informed neural networks, contrasting their approach with previous similar works and traditional numerical methods. We find that selecting improved optimization algorithms significantly enhances the accuracy of the results. Simple modifications to the loss function may also improve precision, offering an additional avenue for enhancement. Despite optimization algorithms having a greater impact on convergence than adjustments to the loss function, practical considerations often favor tweaking the latter due to ease of implementation. On a global scale, the integration of an enhanced optimizer and a marginally adjusted loss function enables a reduction in the loss function by several orders of magnitude across diverse physical problems. Consequently, our results obtained using compact networks (typically comprising 2 or 3 layers of 20-30 neurons) achieve accuracies comparable to finite difference schemes employing thousands of grid points. This study encourages the continued advancement of PINNs and associated optimization techniques for broader applications across various fields.}
}
