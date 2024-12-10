# Quasi-Newton optimization algorithms to enhance PINNs

This repository includes the implementation of the Self-scaled Quasi-Newton algorithms suggested in:
* [Unveiling the optimization process of Physics Informed Neural Networks: How accurate and competitive can PINNs be?](https://www.sciencedirect.com/science/article/pii/S0021999124009045)

together with numerous scripts corresponding to the examples discussed in this work. These scripts contain numerous comments, in order to facilitate their use and modification. Other optimization algorithms will also be included in the future.

# How to use

In order to use the codes associated with each of the examples, you should first download the codes “modified_minimize.py” and “modified_optimize.py”, and then replace respectively the scripts “_minimize.py” and “_optimize.py” by these two files. You will find the scripts “_minimize.py” and “_optimize.py” in the "optimize" folder where you have Scipy installed.

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
