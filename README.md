# Multiscale Cholesky Preconditioning
This is a customized version of mschol implementation.
Courtesy to the original author https://gitlab.inria.fr/geomerix/ichol.

Additional features: edge-preserved smoothing. Since the LHS system is regular, AMG solver works better. (Experiments: AMG <30 iters to converge, ichol preconditioner: around 2k iters to converge)

## ToDO: test performance on heterogeneous data, elastic problem
## Install dependencies

This project depends on several third-party libraries. Besides some
header-only libaries put in `external/`, remaining ones can be easily
installed via the package manager on a Linux desktop.

    sudo apt install libboost-all-dev libgmp-dev libopenblas-dev

If Intel
CPUs are available, [Intel
MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html#gs.4449n7)
are recommended for the implementation of `blas` and `lapack`
subroutines.

For windows users, one can install [GMP library](https://gmplib.org/)
via `vcpkg` package manager, or directly download precompiled binaries
instead from [here](https://github.com/CGAL/cgal/releases). 

## How to build
    
    mkdir build; cd build; cmake -DCMAKE_BUILD_TYPE=Release ..; make -j8 -k;

## Run examples
    
    cd makefiles; ./run_problem_3d
