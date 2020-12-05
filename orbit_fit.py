#usage: mega_orbit_fit.py <number_of_steps>
#must be run in the same folder as sgr_input_data.npy and vc_rh_arrays_log.npy

import time
import sys
import os
from astropy.io import ascii
from astropy.coordinates import ICRS, Galactic, SkyCoord, Distance
from astropy import units as u
import astropy.coordinates as coord
import gala.coordinates as gc
from gala.units import galactic
from scipy.interpolate import interp1d
import scipy
import gala
import astropy.constants as ac
import emcee
import corner
import gala.coordinates as gc
import matplotlib.cm as cm
import numpy as np
from numpy import *

sys.path.append(os.getcwd()) 

# 2020 note: I'm preserving the comments of my deranged former self for posterity

walkerSteps = sys.argv[1]

# what's the time?
start = time.time()
timename = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

# BRING ME DATA!
table = np.load('sgr_input_data.npy')
v_cArray, r_hArray = np.load('vc_rh_arrays_log.npy')
rh_interp = interp1d(v_cArray, r_hArray)
outname = 'sgr_{0}_steps_{1}.npy'.format(walkerSteps, timename)
SgrCore = SkyCoord("18h55m19.5s -30d32m43s", frame = ICRS).galactic

# LET THERE BE GALAXY!
pot = gala.potential.CompositePotential()
pot['bulge'] = gala.potential.HernquistPotential(m = 3.4e10, c = 0.7, units = (u.kpc, u.Myr, u.Msun, u.rad))
pot['disk']  = gala.potential.MiyamotoNagaiPotential(m = 1.0e11, a = 6.5, b = 0.26, units = (u.kpc, u.Myr, u.Msun, u.rad))
pot['halo']  = gala.potential.LogarithmicPotential(
    v_c = (73. * np.sqrt(2.) * u.km / u.second).to(u.kpc / u.Myr).value,
    r_h=12.,
    q1=1.,
    q2=1.,
    q3=1.,
    units=(u.kpc, u.Myr, u.Msun, u.rad))

# A most handy thing to have defined
def gaussian(x, mu, sigma_sq):
	coefficient = 1 / np.sqrt(2 * np.pi * sigma_sq)
	exponent = - (x - mu)**2 / (2 * sigma_sq)
	return coefficient * np.exp(exponent)

# something to do with Sohn IC??? Nah fam maybe not

# Newberg ICs BUT NOT
(l, b, R) = (SgrCore.l, SgrCore.b, 28) # distance from Siegel et al, give DM = 17.27. Same figure is used by Law&Majewski

Sgr_ic = SkyCoord(
    l = l,
    b = b,
    distance = R,
    unit = (u.degree, u.degree, u.kpc),
    frame='galactic',
    galcen_distance = 8.0 * u.kpc)

rr_c = SkyCoord(
    ra = table['RA'],
    dec = table['dec'],
    distance = table['d'],
    unit = (u.degree, u.degree, u.kpc),
    galcen_distance = 8.0 * u.kpc)

rr_sgr = rr_c.transform_to(gc.Sagittarius)
rr_sgr.Lambda.wrap_angle = 180 * u.degree

c = coord.Galactocentric(
    [Sgr_ic.cartesian.x - 8 * u.kpc] * u.kpc,
    [Sgr_ic.cartesian.y] * u.kpc,
    [Sgr_ic.cartesian.z] * u.kpc,
    galcen_distance = 8.0 * u.kpc)

Sgr_vxyz = [[230.], [-35.], [195.]] * u.km/u.s 
c_gc = c.transform_to(coord.Galactic)

pm_l, pm_b, vrad = gc.vgal_to_hel(c_gc, Sgr_vxyz, vcirc = 220 * u.km / u.s, vlsr = [9., 12., 7.] * u.km / u.s)
pm_l = pm_l.to(u.mas/u.yr)
pm_b = pm_b.to(u.mas/u.yr)

empty_galframe = coord.Galactocentric(galcen_distance = 8. * u.kpc)

DMs = 5. * np.log10(table['d']) - 5.
DMs_err = np.divide(table['d_err'], table['d']) * 5 / np.log(10)

temp = coord.SkyCoord(ra = table['RA'] * u.deg, dec = table['dec'] * u.deg)
pms_equ = [table['pmra'], table['pmdec']] * u.mas/u.yr
RR_pml, RR_pmb = gc.pm_icrs_to_gal(temp, pms_equ)
err_pms_equ = [table['pmra_err'],table['pmdec_err']] * u.mas/u.yr
eRR_pml, eRR_pmb = np.absolute(gc.pm_icrs_to_gal(temp, err_pms_equ))

def lnlike(
    p,
    nsteps = 1000,
    l0 = SgrCore.l.value,
    ls = rr_sgr.Lambda,
    bs= rr_sgr.Beta,
    DMs = DMs,
    DMs_err = DMs_err,
    RR_pml = RR_pml,
    RR_pmb = RR_pmb,
    eRR_pml = eRR_pml,
    eRR_pmb = eRR_pmb):

    l = l0
    b, DM, pm_l, pm_b, vrad, sb, spml, spmb, sdm, vc = p # pm_l is actually pm_l_cosb!
    try: rh_interp(vc)
    except ValueError:
        return -inf

    pot['halo']  = gala.potential.LogarithmicPotential(
        v_c = (vc * np.sqrt(2.) * u.km / u.second).to(u.kpc / u.Myr).value,
        r_h = rh_interp(vc),
        q1 = 1.,
        q2 = 1.,
        q3 = 1.,
        units=(u.kpc, u.Myr, u.Msun, u.rad))
    
    distance = 10.**((DM/5.)+1.)/1000.

    # next 20 lines are all about getting properties of an orbit integrated from new_cg, which is generated from p
    try: new_cg = coord.Galactic(
        l = l * u.deg,
        b = b * u.deg,
        distance = distance * u.kpc,
        pm_l_cosb = pm_l * u.mas/u.yr,
        pm_b = pm_b * u.mas/u.yr,
        radial_velocity = vrad * u.km/u.s)
    except:
        return -inf
        
    new_cg_galactocentric = new_cg.transform_to(coord.Galactocentric(galcen_distance=8.*u.kpc))

    new_init = [new_cg_galactocentric.x.value, 
            new_cg_galactocentric.y.value, 
            new_cg_galactocentric.z.value, 
            (new_cg_galactocentric.v_x.value/977.7922216731282), 
            (new_cg_galactocentric.v_y.value/977.7922216731282), 
            (new_cg_galactocentric.v_z.value/977.7922216731282)] # dividing by 977 means v ends up in kpc/Myr

    reverse_orbit = pot.integrate_orbit(array(new_init), dt = -1, n_steps = nsteps)
    new_orbit = pot.integrate_orbit(reverse_orbit[-1], dt = 1, n_steps = nsteps * 2)

    orbitg = new_orbit.to_coord_frame(coord.Galactic, galactocentric_frame = empty_galframe) # properties l,b,distance,pml_cosb,pmb,vrad. 2001 sets of coords.
    orbit_sgr = orbitg.transform_to(gc.Sagittarius) # has properties Lambda, Beta, Distance, pmLambda_cosBeta, pmBeta, vrad

    orbit_sgr.Lambda.wrap_angle=180. * u.deg # was previously 360
    f1 = interp1d(orbit_sgr.Lambda, orbit_sgr.Beta)
    f2 = interp1d(orbit_sgr.Lambda, 5. * np.log10(orbit_sgr.distance.value * 1000.) - 5.)
    #f3 = interp1d(orbit_sgr.Lambda, radvels)  # unused
    f4 = interp1d(orbit_sgr.Lambda, orbitg.pm_l_cosb)
    f5 = interp1d(orbit_sgr.Lambda, orbitg.pm_b)

    try: f1(ls)
    except ValueError: 
        return -inf
    try: f2(ls)
    except ValueError: 
        return -inf
    try: f4(ls)
    except ValueError: 
        return -inf
    try: f5(ls)
    except ValueError: 
        return -inf

    N_b = gaussian(bs.value, f1(ls), (sb*np.ones(len(bs)))**2.)
    N_d = gaussian(DMs, f2(ls), DMs_err**2. + sdm**2.)
    N_pml = gaussian(RR_pml.value, f4(ls), eRR_pml.value**2. + spml**2.)
    N_pmb = gaussian(RR_pmb.value, f5(ls), eRR_pmb.value**2. + spmb**2.)
    i_likelihood = N_b * N_d * N_pml * N_pmb

    if np.any(i_likelihood == 0):
        return -inf
    if np.any(i_likelihood == NaN):
        return -inf
    if np.any(i_likelihood == -NaN):
        return -inf

    return np.sum(np.log(abs(i_likelihood)))

def lnprior(p):

    # This entire function is complete nonsense
    # Most of the parameters of p are actually unused (EVEN THOUGH I DESCRIBED THEM IN THE REPORT :|)

    b, DM, pm_l, pm_b, vrad, sb, spml, spmb, sdm, vc = p
    if sb <= 0:
        return -inf
    else: prior_sb = sb**-1
    if spml <= 0:
        return -inf
    else: prior_spml = spml**-1
    if spmb <= 0:
        return -inf
    else: prior_spmb = spmb**-1
    if sdm <= 0:
        return -inf
    else: prior_sdm = sdm**-1

    prior_dm = 10**(2 * DM / 5)
    
    #not really sure what I'm doing in the next few lines
    if -300 < vrad < 300 and 68 < vc < 200:
        return 0.0
    else:
        return -inf

    # lol I hope anything *after* that if-else isn't important
    # Oh wait, what's this, all of the functionality?

    return np.log(prior_sb * prior_spml * prior_spmb * prior_sdm * prior_dm)

def lnprob(p):
    lp = lnprior(p)
    if not np.isfinite(lp):
        return -inf

    return lp + lnlike(p)

p_start = [
    b.value,
    5. * np.log10(28400.) - 5.,
    pm_l.value,
    pm_b.value,
    vrad.value,
    5.,
    1.,
    1.,
    5.,
    100.] # b dispersion might need to be increased # have increased to 5

Nwalker = 144
Ndim = 10
p0 = [p_start+[.2,.05,.1,.1,1.,2.,.2,.2,1.,20.] * random.randn(Ndim) for i in range(Nwalker)]

sampler = emcee.EnsembleSampler(Nwalker, Ndim, lnprob, threads = 16)
pos, prob, state = sampler.run_mcmc(p0, int(sys.argv[1])) # int(sys.argv[1]) is the number of walker steps. Refer to usage.

samples = np.save(outname, (sampler.chain, sampler.flatchain))
finalv = np.median(sampler.flatchain[:,9])
pot['halo']  = gala.potential.LogarithmicPotential(
    v_c = (finalv * np.sqrt(2.) * u.km / u.second).to(u.kpc / u.Myr).value,
    r_h = rh_interp(finalv),
    q1 = 1.,
    q2 = 1.,
    q3 = 1.,
    units = (u.kpc, u.Myr, u.Msun, u.rad))

#calculate the mass enclosed

print (finalv)
print (pot.mass_enclosed([0, 60, 0]))
print (pot['halo'].mass_enclosed([0, 60, 0]))
print (pot['disk'].mass_enclosed([0, 60, 0]))
print (pot['bulge'].mass_enclosed([0, 60, 0]))

end = time.time()
print (end - start)
