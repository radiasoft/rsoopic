import pytest
import math
import numpy as np

import rsoopic.h2crosssections as h2crosssections
from scipy.constants import c

def test_ejected_electron_energy_spectrum():
    """
    Unit test of energy spectrum of electrons ejected from ionization event
    """
    #beta = 0.06247 # for 1 keV kinetic energy
    beta = 0.19499 # for 10 keV kinetic energy
    n = 10000
    v = np.full((n), beta * c)
    energy = h2crosssections.ejectedEnergy(v, n)
    Emean = np.mean(energy)
    Estd = np.std(energy)
    if __name__ == '__main__':
        print Emean, Estd
    np.testing.assert_approx_equal(Emean, 48, 1)
    np.testing.assert_approx_equal(Estd, 190, 1)

if __name__ == '__main__':
    test_ejected_electron_energy_spectrum()
