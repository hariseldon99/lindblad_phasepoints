lindblad_phasepoints
=============

 BBGKY dynamics for open (Lindbladian) quantum spin gases in a radiation field, averaged over phase point operators

Introduction
-----
The Lindblad dyamics that has been coded is 


Here, the Pauli SU(2) representation is used, the Roman indices run through the lattice sites, and Greek indices run through the three spatial directions. A lattice in any dimensions and with any layout can be used, as long as the site index counting is flattened to 1d. Note that the initial condition is hard-coded to a fully x-polarized state.

The relevant terms (from left to right) are:

The module outputs 


The code can be used in a single processor environment, or a multiprocessor grid using [mpi4py](http://mpi4py.scipy.org/),  the Python bindings of the MPI standard.

Installation
-----
Installation involves three steps. Install git, clone this code repository, install python and all dependencies and build/install the python module(s).

1. Installing git: If git is not already installed in your system, [follow the instructions here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). 

2. Cloning this repository: If you are on any unix-like shell environment, whether an actual unix shell (like [bash](https://www.gnu.org/software/bash/) ), a graphical terminal emulator (like [xterm](http://invisible-island.net/xterm/xterm.html), [gnome-terminal](https://help.gnome.org/users/gnome-terminal/stable/), [yakuake](https://yakuake.kde.org/) etc.) on Linux with X-Windows ([Ubuntu](http://www.ubuntu.com/), [Debian](https://www.debian.org/), [OpenSuse](https://www.opensuse.org/en/) etc.) or an environment like [Cygwin](https://www.cygwin.com/) or [MinGW](http://mingw.org/) on Microsoft Windows, just install git if necessary and run
     ```
     $ git clone https://github.com/hariseldon99/lindblad_phasepoints
     $ echo $PWD
     ```
This causes git clone the repository to the path $PWD/dtwa_quantum_systems.     
In other cases, refer to [git setup guide](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup) and 
[git basics](https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository).

3. Install python and dependencies: If python is not already installed in your system, then refer to [these general instructions](https://wiki.python.org/moin/BeginnersGuide/Download) to download and install python and the dependencies given in the 'External dependencies' section below. Alternatively, install a python distribution like [anaconda](https://store.continuum.io/cshop/anaconda/) and use it's internal package management to install the required dependencies.

4. Build and install the python module(s)
    ```
    $ python setup.py build
    $ python setup.py install
    ```
  The first command builds the python module "lindblad_phasepoints", as well as the optimized bbgky. The latter requires a 
  [BLAS](http://www.netlib.org/blas/) library to be installed as a [shared library](http://tldp.org/HOWTO/Program-Library-HOWTO/shared-libraries.html). If that has been done already, and is in default paths given by [ldconfig](http://tldp.org/HOWTO/Program-Library-HOWTO/shared-libraries.html) or [LD_LIBRARY_PATH](http://tldp.org/HOWTO/Program-Library-HOWTO/shared-libraries.html), the build should run correctly. Otherwise, edit the file setup.py and change the variables "blas_path" "blas_headers" to point to the path where the library and headers are installed.

  
  If you do not wish to install in root space, run the install with the --user option. Alternatively, don't run the install (just the build), start a python shell (or write a python script) and run

  ```python
  >>> import sys
  >>> sys.path.append("/path/to/build/directory/")
  >>> import lindblad_phasepoints as ldb
  ```

Usage
-----
Usage examples in python are shown below.

Example 1: Obtaining Documentation
```python
>>> import lindblad_phasepoints as ldb
>>> help(dtwa)
>>> import lindblad_bbgky as lb
>>> help(lb)
```

Relevant Literature:
-----

###Relevant papers:
* [PRM: arXiv:1510.03768 (2015)](http://arxiv.org/abs/1510.03768)
* [Wooters: Annals of Physics 176, 1–21 (1987)](http://dx.doi.org/10.1016/0003-4916%2887%2990176-X)
* [Anatoli : Ann. Phys 325 (2010) 1790-1852](http://arxiv.org/abs/0905.3384)
* [Mauritz: New J. Phys. 15, 083007 (2013)](http://arxiv.org/abs/1209.3697)
* [Schachenmayer: Phys. Rev. X 5 011022 (2015)](http://arxiv.org/abs/1408.4441)

###Relevant docs for the bundled version of mpi4py reduce:
* [GitHub](https://github.com/mpi4py/mpi4py/blob/master/demo/reductions/reductions.py)
* [readthedocs.org](https://mpi4py.readthedocs.org/en/latest/overview.html#collective-communications)
* [Google Groups](https://groups.google.com/forum/#!msg/mpi4py/t8HZoYg8Ldc/-erl6BMKpLAJ)


###External dependencies:
1. mpi4py - MPI for Python

    _\_-MPI (Parallelizes the different samplings of the dtwa)

2. numpy - Numerical Python (Various uses)

3. scipy  - Scientific Python

    _\_-integrate 

    _| \_-odeint (Integrates the BBGKY dynamics of the sampled state)

    _| \_-signal 
    
    _| \_-fftconvolve (Used for calculating spin correlations)

4. tabulate - Tabulate module 
    
    _\_-tabulate (Used for dumping tabular data)

5. The bbgky dynamics module

    _\_-gcc/automake (compilers for the C module)
    
    _\_-cblas - Any BLAS library written in C

###TODO:
1. Evaluate site correlations as functions of time and dump the whole data to parallel hdf5
2. Docs! Docs! Lotsa work on the docs.
3. MIGRATE TO PYTHON 3. CHANGE ALL TABS TO SPACES FOR INDENTATION
