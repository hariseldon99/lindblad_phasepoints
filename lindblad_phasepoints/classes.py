#Class Library
import numpy as np
from numpy.linalg import norm
from consts import *

class ParamData:
    """Class that stores Hamiltonian and lattice parameters
       to be used in each dTWA instance. This class has no
       methods other than the constructor.
    """

    def __init__(self, latsize=11, amplitude=1.0, detuning=0.0, \
      cloud_rad=100.0, kvecs=np.array([0.0,0.0,1.0])):

        """
         Usage:
         p = ParamData(latsize=100, \
                        amplitude=1.0, detuning=0.0)

         All parameters (arguments) are optional.

         Parameters:
         latsize           =  The size of your lattice as an integer. This can be in
                                any dimensions
         amplitude  =  The periodic (cosine) drive amplitude
                                Defaults to 1.0.
         detuning          =  The periodic (cosine) drive frequency, i.e.
                                detuning between atomic levels and incident light.
                                Defaults to 0.0.
         cloud_rad  =  The radius of the gas cloud of atoms. Defaults to 100.0
         kvecs      =  Array of 3-vectors. Each vector is a momentum of the emerging radiation
                       field. Defaults to np.array([0.0,0.0,1.0])

         Return value:
         An object that stores all the parameters above.
         Note that the momentum (magnitude) of theincident light
         is scaled to unity and propagates in the z-direction
         by default unless you set the azimuth theta to a specific value
        """
        self.flipped = False
        self.latsize = latsize
        self.drv_amp, self.drv_freq = amplitude, detuning
        self.cloud_rad = cloud_rad
        #Incident laser has unit magnitude in the z-direction
        self.kvec_incident = np.array([0.0, 0.0, 1.0])
        #Set the momenta to be unit magnitude
        self.kvecs = \
          kvecs/np.apply_along_axis(norm, -1, kvecs).reshape(-1,1)
        self.cloud_density = \
          self.latsize/((4./3.) * np.pi * pow(self.cloud_rad,3.0))
        self.intpt_spacing = 0.5/pow(self.cloud_density,1./3.)

class Atom:
    """
    Class that stores all the data specifying a single atom
    in the cloud (index and coordinates).
    Also has methods for extracting the data and calculating
    distance between 2 atoms
    """

    def __init__(self, coords=np.array([0.0,0.0]), index=0):
        """
         Usage:
         import numpy as np
         c = np.array([1.2,0.4])
         a = Atom(coords = c, index = 3)

         All arguments are optional.

         Parameters:
         coords   =  The 2D coordinates of the atom in the cloud, entered as a
                              numpy array of double precision floats np.array([x,y])

         index    =  The index of the atom while being counted among others.
                              These are counted from 0

         Extra Member
         state            = The current state of the atom in the BBGKY Heirarchy
                             This is a numpy array of size 3*N + 9*N^2, where N is
                             the total # of atoms. Defaults to nalphas array of None.

         Return value:
         An atom object
        """
        if(coords.size == 3):
            self.index = index
            self.coords = coords
            #Initialize with a blank reference state
            #These states will be the local initial state for noneqm spectra
            #These states will be just the r_alphas for eqm spectra
            self.refstate = np.array([None for i in xrange(nalphas)])
            #Initialize with a blank state for each alpha
            self.state = np.array([None for i in xrange(nalphas)])
        else:
            raise ValueError('Incorrect 3D coordinates %d' % (coords))
