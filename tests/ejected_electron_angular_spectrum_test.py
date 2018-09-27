import pytest
import numpy as np

import rsoopic.h2crosssections as h2crosssections

def test_ejected_electron_energy_spectrum():
    """
    Unit test of energy spectrum of electrons ejected from ionization event
    """
    n = 10000
    energy = np.full((n), 10.e3) # incident energy in eV

    angle = 180. / np.pi * h2crosssections.generateAngle(n, 0.5 * energy, energy)
    Amean = np.mean(angle); Astd = np.std(angle)
    if __name__ == '__main__':
        print Amean, Astd
    np.testing.assert_approx_equal(Amean, 45.4, 2)
    np.testing.assert_approx_equal(Astd, 8.8, 1)

    angle = 180. / np.pi * h2crosssections.generateAngle(n, 0.4 * energy, energy)
    Amean = np.mean(angle); Astd = np.std(angle)
    if __name__ == '__main__':
        print Amean, Astd
    np.testing.assert_approx_equal(Amean, 51.1, 2)
    np.testing.assert_approx_equal(Astd, 8.8, 1)

    angle = 180. / np.pi * h2crosssections.generateAngle(n, 0.01 * energy, energy)
    Amean = np.mean(angle); Astd = np.std(angle)
    if __name__ == '__main__':
        print Amean, Astd
    np.testing.assert_approx_equal(Amean, 84.6, 2)
    np.testing.assert_approx_equal(Astd, 21.7, 1)

    angle = 180. / np.pi * h2crosssections.generateAngle(n, 4. * energy, 10. * energy)
    Amean = np.mean(angle); Astd = np.std(angle)
    if __name__ == '__main__':
        print Amean, Astd
    np.testing.assert_approx_equal(Amean, 49.6, 2)
    np.testing.assert_approx_equal(Astd, 4.7, 1)

if __name__ == '__main__':
    test_ejected_electron_energy_spectrum()
