#!/usr/bin/env python
"""
Utility module for calculating ionization cross-sections for H2, a la:

[1] Yong-Ki Kim, Jose Paulo Santos, and Fernando Parente,
    "Extension of the binary-encounter-dipole model to relativistic
    incident electrons", Phys. Rev. A 62, 052710, 13 October 2000
    <http://teddy.physics.utah.edu/papers/physrev/PRA52710.pdf>

[2] D. Bruhwiler, "RNG Calculations for Scattering in XOOPIC", Tech-X Note, 2000
"""
from __future__ import division
import math
import random
import time
import numpy as np
# any import from warp will trigger arg parsing, which will fail without this
# in a notebook context (or anything else with non-warp commandline args)
import warpoptions
warpoptions.ignoreUnknownArgs = True
from warp import emass, clight, jperev
from scipy.constants import physical_constants, fine_structure # fine-structure constant
a_0 = physical_constants['Bohr radius'][0]
r_0 = physical_constants['classical electron radius'][0]
emassEV = emass*clight**2/jperev

I = 15.42593  # threshold ionization energy (in eV), from the NIST Standard Reference Database (via NIST Chemistry WebBook)
U = 15.98 # average orbital kinetic energy (in eV) (value taken from https://physics.nist.gov/PhysRefData/Ionization/intro.html)
R = 13.60569  # Rydberg energy (in eV)
N = 2  # number of electrons in target (H2)

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
    Cross section sigma is given by Eq. (22) in Ref. [1]
    """
    t = normalizedKineticEnergy(vi)
    if t <= 1:
        return 0.

    beta = lambda E: math.sqrt(1. - 1. / (1. + E / emassEV)**2)

    # initialize needed variables
    beta_t = vi / clight
    beta_u = beta(U)
    beta_b = beta(I)
    bprime = I / emassEV
    tprime = t * bprime
    if not useMollerApproximation:
        # now add terms to sigma, one by one:
        sigma = math.log(beta_t**2 / (1. - beta_t**2)) - beta_t**2 - math.log(2. * bprime)
        #print sigma, t, tprime
        sigma *= .5 * (1. - 1. / (t * t))
        #print sigma
        sigma += 1. - 1. / t - math.log(t) / (t + 1.) * (1. + 2. * tprime) / (1. + .5 * tprime)**2
        #print sigma
        sigma += bprime**2 / (1. + .5 * tprime)**2 * (t - 1) / 2.
        #print sigma
        sigma *= 4. * np.pi * a_0**2 * fine_structure**4 * N / (beta_t**2 + beta_u**2 + beta_b**2) / (2. * bprime)
        #print sigma
    else:
        #sigma = 1. - 1. / t - math.log(t) / (t + 1.) * (1. + 2. * tprime) / (1. + tprime)**2
        #sigma += bprime**2 / (1. + tprime)**2 * (t - 1.) / 2.
        #sigma *= 4. * np.pi * a_0**2 * fine_structure**2 * N * (R / I) / (beta_t * beta_t)
        mec2 = emass * clight**2
        kappa = 2. * math.pi * r_0**2 * mec2 / (beta_t * beta_t)
        eps_min = 10. * jperev
        eps = tprime * mec2
        sigma = 1. / eps_min - 1. / (eps - eps_min)
        sigma += (eps - 2. * eps_min) / (2. * (mec2 + eps)**2)
        sigma += mec2 * (mec2 + 2. * eps) / (eps * (mec2 + eps)**2) \
        math.log(eps_min / (eps - eps_min))
        sigma *= kappa

    return sigma


def ejectedEnergy(vi, nnew):
    """
    Selection of an ejected electron energy (in eV), adapted from
    XOOPIC's MCCPackage::ejectedEnergy routine
    """
    vi = vi[0:nnew]  # We may be given more velocities than we actually need

    tstart = time.time()
    gamma_inc = 1/np.sqrt(1-(vi/clight)**2)
    impactEnergy = (gamma_inc-1) * emassEV

    if useMollerApproximation:
        eps_min = 10.
        wOutMoller = np.empty((nnew))
        for i in range(nnew):
            Xrand = random.random()
            wOutMoller[i] = impactEnergy[i] * eps_min / (impactEnergy[i] - Xrand * (impactEnergy[i] - 2 * eps_min))
        return wOutMoller

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

    This inversion is explained in detail in [2].
    """

    if useMollerApproximation:
        thetaMoller = np.empty((nnew))
        for i in range(nnew):
            costheta = emitted_energy[i] * (incident_energy[i] + 2. * emassEV)
            costheta /= incident_energy[i] * (emitted_energy[i] + 2. * emassEV)
            costheta = math.sqrt(costheta)
            thetaMoller[i] = math.acos(costheta)
        return thetaMoller

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


def normalizedKineticEnergy(vi=None):
    """
    Compute the normalized kinetic energy n = T/I given an input velocity
    """
    gamma_in = 1. / np.sqrt(1 - (vi/clight)**2)
    T = (gamma_in - 1) * emass * clight**2 / jperev  # kinetic energy (in eV) of incident electron
    t = T / I  # normalized kinetic energy
    return t
