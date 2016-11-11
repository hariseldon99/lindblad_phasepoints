from distutils.core import setup, Extension
import numpy as np
import os
#Change these as needed
blas_dir = os.environ['HOME'] + "/.local/"
blas_path = blas_dir + 'lib'
blas_headers = blas_dir + 'include'
#Order of optimization or any other compiler options you wanna add
opt = "-O3"
lindblad_bbgky = Extension('lindblad_bbgky',\
  include_dirs = [np.get_include(),blas_headers],\
   extra_compile_args = [opt],\
    extra_link_args = [opt],\
    libraries = ['openblas'],\
      library_dirs = [blas_path],\
        sources = ['lindblad_bbgky/bbgkymodule.c', \
          'lindblad_bbgky/lindblad_bbgky.c'])

setup (name = 'lindblad_phasepoints',
        version = '1.0',
        description = """BBGKY dynamics for open (Lindbladian) quantum
                          spin gases in a radiation field, averaged over
                          phase point operators""",
        long_description=\
          """
              BBGKY dynamics for open (Lindbladian) quantum
              spin gases in a radiation field, averaged over
              phase point operators
            * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            * Copyright (c) 2015 Analabha Roy (daneel@utexas.edu)
            *
            *This is free software: you can redistribute it and/or modify
            *it under the terms of version 2 of the GNU Lesser General
            *Public License as published by the Free Software Foundation.
            *Notes:
            *1. The initial state is currently hard coded to be the
            *classical ground  state
            *2. Primary references are
            *   PRM:  arXiv:1510.03768
            *   Schachenmayer: arXiv:1408.4441
            * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         """,
        url='https://github.com/hariseldon99/lindblad_phasepoints',
         # Author details
        author='Analabha Roy',
        author_email='daneel@utexas.edu',
        package_data={'': ['LICENSE']},
        include_package_data=True,

        # Choose your license
        license='GPL',

        # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
        classifiers=[
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 3 - Alpha',

            # Indicate who your project is intended for
            'Intended Audience :: Physicists',
            'Topic :: Numerical Quantum Simulations :: Dynamics',

            # Pick your license as you wish (should match "license" above)
            'License :: GPL License',

            # Specify the Python versions you support here. In particular,
            #ensure that you indicate whether you support Python 2,
            # Python 3 or both.
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.6',
            'Programming Language :: Python :: 2.7',
        ],
        packages=['lindblad_phasepoints','lindblad_bbgky'],
        ext_modules = [lindblad_bbgky])
