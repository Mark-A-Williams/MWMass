# usage: mega_orbit_fit.py <number_of_steps>
# must be run in the same folder as sgr_input_data.npy and vc_rh_arrays_log.npy

import time
import sys
import os
from astropy.io import ascii
from astropy.coordinates import ICRS, Galactic, SkyCoord, Distance
from astropy import units as u
import astropy.coordinates as coord
import gala.coordinates as gc
from gala.potential import CompositePotential
from gala.units import galactic
from scipy.interpolate import interp1d
import scipy
import gala
import astropy.constants as ac
import emcee
import matplotlib.cm as cm
import numpy as np
from tenDParameters import tenDParameters
from probabilities import lnlike, lnprior
sys.path.append(os.getcwd())

def getBasePotential() -> CompositePotential:
    pot = CompositePotential()
    pot['bulge'] = gala.potential.HernquistPotential(m = 3.4e10, c = 0.7, units = (u.kpc, u.Myr, u.Msun, u.rad))
    pot['disk']  = gala.potential.MiyamotoNagaiPotential(m = 1.0e11, a = 6.5, b = 0.26, units = (u.kpc, u.Myr, u.Msun, u.rad))
    return pot

def getRandomisedStartParameters(p: tenDParameters) -> tenDParameters:
    return tenDParameters(
        p.b + 0.2 * np.random.randn(),
        p.DM + 0.05 * np.random.randn(),
        p.pm_l + 0.1 * np.random.randn(),
        p.pm_b + 0.1 * np.random.randn(),
        p.vrad + np.random.randn(),
        p.sb + 2 * np.random.randn(),
        p.spml + 0.2 * np.random.randn(),
        p.spmb + 0.2 * np.random.randn(),
        p.sdm + np.random.randn(),
        p.vc + 20 * np.random.randn())

def lnprob(p):
    lp = lnprior(p)
    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(
        p,
        potential, # this will need updating, as will the next one, since currently they'll just mutated in the method
        rh_interp,
        l0 = sgrCoreFromICRS.l.value,
        ls = rr_sgr.Lambda,
        bs= rr_sgr.Beta,
        DMs = DMs,
        DMs_err = DMs_err,
        RR_pml = RR_pml,
        RR_pmb = RR_pmb,
        eRR_pml = eRR_pml,
        eRR_pmb = eRR_pmb)

walkerSteps = sys.argv[1]

start = time.time()
timename = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

# BRING ME DATA!
rrLyraeData = np.load('sgr_input_data.npy')
v_cArray, r_hArray = np.load('vc_rh_arrays_log.npy')
rh_interp = interp1d(v_cArray, r_hArray)
outname = 'sgr_{0}_steps_{1}.npy'.format(walkerSteps, timename)

potential = getBasePotential()

sgrCoreFromICRS = SkyCoord("18h55m19.5s -30d32m43s", frame = ICRS).galactic
# distance from Siegel et al, give DM = 17.27. Same figure is used by Law&Majewski
sgrCoreInitialCoords = SkyCoord(
    l = sgrCoreFromICRS.l,
    b = sgrCoreFromICRS.b,
    distance = 28,
    unit = (u.degree, u.degree, u.kpc),
    frame='galactic',
    galcen_distance = 8.0 * u.kpc)

rrLyraeCoordinates = SkyCoord(
    ra = rrLyraeData['RA'],
    dec = rrLyraeData['dec'],
    distance = rrLyraeData['d'],
    unit = (u.degree, u.degree, u.kpc),
    galcen_distance = 8.0 * u.kpc)

rr_sgr = rrLyraeCoordinates.transform_to(gc.Sagittarius)
rr_sgr.Lambda.wrap_angle = 180 * u.degree

sgrCoreGalactocentric = coord.Galactocentric(
    [sgrCoreInitialCoords.cartesian.x - 8 * u.kpc] * u.kpc,
    [sgrCoreInitialCoords.cartesian.y] * u.kpc,
    [sgrCoreInitialCoords.cartesian.z] * u.kpc,
    galcen_distance = 8.0 * u.kpc)

sgrCoreVelocity = [[230.], [-35.], [195.]] * u.km/u.s 
sgrCoreGalactic = sgrCoreGalactocentric.transform_to(coord.Galactic)

# 2020 NOTE
# https://github.com/adrn/gala/commit/23b0babcd1ad76b01a1cd1712fb1425f786cb641#diff-f041e9f9bedf76d31f62d6ea190a80c11929fece544f7c4d0d4a11d551a5c688
# vgal_to_vhel deprecated, use an astropy thing
# how to initialise a skycoord with the cartesian sgrCoreVelocity??
sgrCorePM_l, sgrCorePM_b, sgrCoreVRad = gc.vgal_to_hel(
    sgrCoreGalactic,
    sgrCoreVelocity,
    vcirc = 220 * u.km / u.s,
    vlsr = [9., 12., 7.] * u.km / u.s)

sgrCorePM_l = sgrCorePM_l.to(u.mas/u.yr)
sgrCorePM_b = sgrCorePM_b.to(u.mas/u.yr)

DMs = 5. * np.log10(rrLyraeData['d']) - 5.
DMs_err = np.divide(rrLyraeData['d_err'], rrLyraeData['d']) * 5 / np.log(10)

temp = coord.SkyCoord(ra = rrLyraeData['RA'] * u.deg, dec = rrLyraeData['dec'] * u.deg)
pms_equ = [rrLyraeData['pmra'], rrLyraeData['pmdec']] * u.mas/u.yr
RR_pml, RR_pmb = gc.pm_icrs_to_gal(temp, pms_equ)
err_pms_equ = [rrLyraeData['pmra_err'],rrLyraeData['pmdec_err']] * u.mas/u.yr
eRR_pml, eRR_pmb = np.absolute(gc.pm_icrs_to_gal(temp, err_pms_equ))

# set it up

fixedStartParameters = tenDParameters(
    b = sgrCoreFromICRS.b.value,
    DM = 5 * np.log10(28400) - 5,
    pm_l = sgrCorePM_l.value,
    pm_b = sgrCorePM_b.value,
    vrad = sgrCoreVRad.value,
    sb = 5,
    spml = 1,
    spmb = 1,
    sdm = 5,
    vc = 100) # this value seems to be in some contention. I claimed 130 in the report...

numberOfWalkers = 144
p0 = [getRandomisedStartParameters(fixedStartParameters) for i in range(numberOfWalkers)]

sampler = emcee.EnsembleSampler(numberOfWalkers, 10, lnprob, threads = 16)
pos, prob, state = sampler.run_mcmc(p0, walkerSteps)

samples = np.save(outname, (sampler.chain, sampler.flatchain))
finalv = np.median(sampler.flatchain[:,9])
potential['halo']  = gala.potential.LogarithmicPotential(
    v_c = (finalv * np.sqrt(2) * u.km / u.second).to(u.kpc / u.Myr).value,
    r_h = rh_interp(finalv),
    q1 = 1.,
    q2 = 1.,
    q3 = 1.,
    units = (u.kpc, u.Myr, u.Msun, u.rad))

#calculate the mass enclosed

print (finalv)
print (potential.mass_enclosed([0, 60, 0]))
print (potential['halo'].mass_enclosed([0, 60, 0]))
print (potential['disk'].mass_enclosed([0, 60, 0]))
print (potential['bulge'].mass_enclosed([0, 60, 0]))

end = time.time()
print (end - start)
