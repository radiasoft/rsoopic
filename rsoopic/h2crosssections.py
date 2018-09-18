#!/usr/bin/env python
"""
Utility module for calculating ionization cross-sections for H2, a la:

[1] Yong-Ki Kim, José Paulo Santos, and Fernando Parente,
    "Extension of the binary-encounter-dipole model to relativistic
    incident electrons", Phys. Rev. A 62, 052710, 13 October 2000
    <http://teddy.physics.utah.edu/papers/physrev/PRA52710.pdf>

[2] D. Bruhwiler, "RNG Calculations for Scattering in XOOPIC", Tech-X Note, 2000

"""
from __future__ import division
import time
import numpy as np
# any import from warp will trigger arg parsing, which will fail without this
# in a notebook context (or anything else with non-warp commandline args)
import warpoptions
warpoptions.ignoreUnknownArgs = True
from warp import emass, clight, jperev

emassEV = emass*clight**2/jperev
bohr_radius = 5.29177e-11  # Bohr Radius (in m)
I = 15.42593  # Threshold ionization energy (in eV), from the NIST Standard Reference Database (via NIST Chemistry WebBook)
R = 13.60569  # Rydberg energy (in eV)
N = 2  # number of electrons in target (H2)
S = 4 * np.pi * bohr_radius**2 * N * (R/I)**2
fitparametern = 2.4  # species-dependent fitting parameter

useMollerApproximation = False

def n2_ioniz_ddcs(vi=None, vo=None, theta=None):
    """
    Compute the doubly-differential cross-section (not implemented)
    """
    return 0.


def n2_ioniz_sdcs(vi=None, vo=None):
    """
    Compute the differential cross-section (not implemented)
    """
    return 0.


def h2_ioniz_crosssection(vi=None):
    """
    Compute the total cross-section for impact ionization of H2 by e-
    vi - incident electron velocity in m/s; this is passed in from warp as vxi=uxi*gaminvi etc.
    """
    t = normalizedKineticEnergy(vi)

    n = fitparametern

    def g1(t, n): # Eq. 7 in Ref. [1]
        return (1 - t**(1-n)) / (n-1) - (2 / (t+1))**(n/2) * (1 - t**(1 - n/2)) / (n-2)

    sigma = S * F(t) * g1(t, n) # Eq. 6 in Ref. [1]
    return np.nan_to_num(sigma)


def ejectedEnergy(vi, nnew):
    """
    Selection of an ejected electron energy (in eV), adapted from
    XOOPIC's MCCPackage::ejectedEnergy routine
    """
    vi = vi[0:nnew]  # We may be given more velocities than we actually need

    tstart = time.time()
    gamma_inc = 1/np.sqrt(1-(vi/clight)**2)
    impactEnergy = (gamma_inc-1) * emassEV

    tPlusMC = impactEnergy + emassEV
    twoTplusMC = impactEnergy + tPlusMC
    tPlusI = impactEnergy + I
    tMinusI = impactEnergy - I

    invTplusMCsq = 1. / (tPlusMC * tPlusMC)
    tPlusIsq = tPlusI ** 2
    iOverT = I / impactEnergy

    funcT1 = 14./3. + .25 * tPlusIsq * invTplusMCsq - emassEV * twoTplusMC * iOverT * invTplusMCsq
    funcT2 = 5./3. - iOverT - 2.*iOverT*iOverT/3. + .5 * I * tMinusI * invTplusMCsq + emassEV*twoTplusMC*I*invTplusMCsq*np.log(iOverT)/tPlusI

    aGreaterThan = funcT1 * tMinusI / funcT2 / tPlusI

    needToTryAgain = True
    npart = nnew  # number of particles to generate on each loop
    frand = np.random.uniform

    wOut = np.array([])

    while len(wOut) < nnew:
        # print("%i particles left to generate for" % npart)

        # wTest is the inverse of F(W)
        # The random number, called F(W) in the notes and randomFW here,
        #   is the antiderivative of a phony_but_simple probability, which
        #   is always >= the correct_but_messy probability.

        randomFW = frand(size=nnew) * aGreaterThan

        # wTest is the inverse of F(W)
        wTest = I*funcT2*randomFW/(funcT1-funcT2*randomFW)

        # Because we are not working directly with the distribution function,
        #   we must use the "rejection method".  This involves generating
        #   another random number and seeing whether it is > the ratio of
        #   the true probability over the phony_but_simple probability.

        wPlusI       = wTest + I
        wPlusIsq     = wPlusI * wPlusI
        invTminusW   = 1./(impactEnergy-wTest)
        invTminusWsq = invTminusW**2
        invTminusW3  = invTminusW**3

        probabilityRatio = (1. + 4.*I/wPlusI/3. + wPlusIsq*invTminusWsq + 4.*I*wPlusIsq*invTminusW3/3. - emassEV*twoTplusMC*wPlusI*invTminusW*invTplusMCsq + wPlusIsq*invTplusMCsq) / funcT1

        mask = (probabilityRatio >= frand(size=nnew))
        # npart -= np.sum(mask)  # Decrement by the number of passing particles

        # Append the energies that meet the selection criterion
        wOut = np.append(wOut, wTest[mask])

    print("Spent %.3f s generating ejected energies" % (time.time()-tstart))
    return wOut[0:nnew]  # Might possibly have more particles than necessary, but should have at least that many


def generateAngle(nnew, emitted_energy, incident_energy):
    """
    emitted_energy - emitted electon energy (in eV)
    incident_energy - incident electon energy (in eV)
    nnew - number of new particles

    Selection of after-ionization angles for the primary and secondary electrons
    adapted from XOOPIC's MCCPackage::primarySecondaryAngles routine.

    ## The general idea ##
    The cross-section $\sigma(w, t, \theta)$ dictates the likelihood
    of any given ionization event with incident energy $t$, emission energy $w$, and
    emission angle $\theta$.  We will treat the recoiling primary as though it is
    two separate particles, one before and one after ionization, with energy
    reduced by the ionization energy and what is given to the secondary.

    Note: theta is defined relative to the incident electron trajectory

    We know that:
        $ \int_0^\pi { \sigma(w, t, \theta) * 2\pi sin(\theta) d\theta } = \sigma(t,w) $

    We can then define $F(\theta)$ taking the value $0 <= F(\theta) <= 1$:
        $ F(\theta) = 2\pi \int_0^\theta d\theta' sin(\theta') \frac{\sigma(w, t, \theta')}{\sigma(w,t)} $

    If we can invert this expression to get $\theta$ as a function of $w, t$, and $F(\theta)$, we can
    sample this distribution by choosing a random number for $F(\theta)$ and the known values of $w, t$.

    This inversion is explained in detail in [1].
    """

    T = incident_energy
    W = emitted_energy
    g2 = G_2(T, W)
    g3 = G_3(T, W)
    F = np.random.uniform(size=nnew)

    theta = np.nan_to_num(np.arccos(g2 + g3 * np.tan(
        (1-F) * np.arctan2((1-g2), g3) - F * np.arctan2((1+g2), g3))
    ))
    return theta


def alpha(E):
    """
    E - electron energy (in eV)
    """
    return 0.6 * (emassEV / (E + emassEV))**2


def G_2(T, W):
    """
    T - incident electron energy (in eV)
    W - emitted electon energy (in eV)
    """
    return np.sqrt(np.divide(W+I, T) * np.divide(T+2*emassEV, W+2*emassEV))


def G_3(T, W):
    """
    T - incident electron energy (in eV)
    W - emitted electon energy (in eV)
    """
    return alpha(T) * np.sqrt(np.divide(I, W) * np.divide(T - (W+I), T))


def F(t):
    """
    The Bethe-like function in Rudd's modification of the Mott equation.
    See [1] eq. 9
    """
    # Parameters for H2 specific to this fit
    a1 = 0.74
    a2 = 0.87
    a3 = -0.60
    return np.divide(1, t) * (a1 * np.log(t) + a2 + a3 * np.divide(1, t))


def f_1(w, t, n=fitparametern):
    """
    The Mott-like expression in Rudd's cross-section, see [1] eq. 4
    """
    return np.divide(1, (w+1)**n) + np.divide(1, (t-w)**n) - np.divide(1, ((w+1) * (t-w))**(n/2.))


def normalizedKineticEnergy(vi=None):
    """
    Compute the normalized kinetic energy n = T/I given an input velocity
    """
    gamma_in = 1. / np.sqrt(1 - (vi/clight)**2)
    T = (gamma_in - 1) * emass * clight**2 / jperev  # kinetic energy (in eV) of incident electron
    t = T / I  # normalized kinetic energy
    return t
