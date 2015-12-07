# Author:  Analabha roy
# Contact: daneel@utexas.edu
from __future__ import division, print_function
"""
   BGKY dynamics for open (Lindbladian) quantum 
   spin gases in a radiation field, averaged over phase point operators
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
"""

__version__   = '0.1'
__author__    = 'Analabha Roy'
__credits__   = 'Lorenzo Pucci, NiTheP Stellenbosch'

__all__ = ["consts", "classes", "bbgky_noneqm"]
from bbgky_noneqm import *
