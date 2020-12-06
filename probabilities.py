from tenDParameters import tenDParameters
import numpy as np
import astropy.coordinates as coord
import astropy.units as u
from scipy.interpolate import interp1d
import gala
import gala.coordinates as gc
from gala.potential import CompositePotential

def gaussian(x, mu, sigma_sq):
	coefficient = 1 / np.sqrt(2 * np.pi * sigma_sq)
	exponent = - (x - mu)**2 / (2 * sigma_sq)
	return coefficient * np.exp(exponent)

def lnlike(p: tenDParameters, potential: CompositePotential, rh_interp: interp1d, l0, ls, bs, DMs, DMs_err, RR_pml, RR_pmb, eRR_pml, eRR_pmb) -> float:

    try: rh_interp(p.vc)
    except ValueError:
        return -np.inf

    potential['halo']  = gala.potential.LogarithmicPotential(
        v_c = (p.vc * np.sqrt(2) * u.km / u.second).to(u.kpc / u.Myr).value,
        r_h = rh_interp(p.vc),
        q1 = 1.,
        q2 = 1.,
        q3 = 1.,
        units=(u.kpc, u.Myr, u.Msun, u.rad))
    
    distance = 10.**((p.DM / 5.) + 1.) / 1000.

    # next 20 lines are all about getting properties of an orbit integrated from new_cg, which is generated from p
    try: currentTestCoords = coord.Galactic(
        l = l0 * u.deg,
        b = p.b * u.deg,
        distance = distance * u.kpc,
        pm_l_cosb = p.pm_l * u.mas/u.yr,
        pm_b = p.pm_b * u.mas/u.yr,
        radial_velocity = p.vrad * u.km/u.s)
    except:
        return -np.inf
        
    testCoordsGalactocentric = currentTestCoords.transform_to(coord.Galactocentric(galcen_distance = 8. * u.kpc))
    velocityConversion = 977.7922216731282 # dividing by 977 means v ends up in kpc/Myr
    orbitStart = [testCoordsGalactocentric.x.value, 
            testCoordsGalactocentric.y.value, 
            testCoordsGalactocentric.z.value, 
            (testCoordsGalactocentric.v_x.value / velocityConversion), 
            (testCoordsGalactocentric.v_y.value / velocityConversion), 
            (testCoordsGalactocentric.v_z.value / velocityConversion)]

    nsteps = 1000
    reverse_orbit = potential.integrate_orbit(np.array(orbitStart), dt = -1, n_steps = nsteps)
    new_orbit = potential.integrate_orbit(reverse_orbit[-1], dt = 1, n_steps = nsteps * 2)

    empty_galframe = coord.Galactocentric(galcen_distance = 8. * u.kpc)
    orbitg = new_orbit.to_coord_frame(coord.Galactic, galactocentric_frame = empty_galframe) # properties l,b,distance,pml_cosb,pmb,vrad. 2001 sets of coords.
    orbit_sgr = orbitg.transform_to(gc.Sagittarius) # has properties Lambda, Beta, Distance, pmLambda_cosBeta, pmBeta, vrad

    orbit_sgr.Lambda.wrap_angle = 180. * u.deg # was previously 360
    f1 = interp1d(orbit_sgr.Lambda, orbit_sgr.Beta)
    f2 = interp1d(orbit_sgr.Lambda, 5. * np.log10(orbit_sgr.distance.value * 1000) - 5)
    #f3 = interp1d(orbit_sgr.Lambda, radvels)  # unused
    f4 = interp1d(orbit_sgr.Lambda, orbitg.pm_l_cosb)
    f5 = interp1d(orbit_sgr.Lambda, orbitg.pm_b)

    try: f1(ls)
    except ValueError: 
        return -np.inf
    try: f2(ls)
    except ValueError: 
        return -np.inf
    try: f4(ls)
    except ValueError: 
        return -np.inf
    try: f5(ls)
    except ValueError: 
        return -np.inf

    N_b = gaussian(bs.value, f1(ls), (p.sb * np.ones(len(bs)))**2)
    N_d = gaussian(DMs, f2(ls), DMs_err**2 + p.sdm**2)
    N_pml = gaussian(RR_pml.value, f4(ls), eRR_pml.value**2 + p.spml**2)
    N_pmb = gaussian(RR_pmb.value, f5(ls), eRR_pmb.value**2 + p.spmb**2)
    i_likelihood = N_b * N_d * N_pml * N_pmb

    if np.any(i_likelihood == 0):
        return -np.inf
    if np.any(i_likelihood == np.NaN):
        return -np.inf
    if np.any(i_likelihood == -np.NaN):
        return -np.inf

    return np.sum(np.log(abs(i_likelihood)))

def lnprior(p: tenDParameters) -> float:
    # Prior for vrad does not line up with what I put in the report...
    if not (-300 < p.vrad < 300) or not (68 < p.vc < 200):
        return -np.inf

    # No prior for b
    # No priors for pml or pmb

    if p.sb <= 0:
        return -np.inf
    else: prior_sb = p.sb**-1
    if p.spml <= 0:
        return -np.inf
    else: prior_spml = p.spml**-1
    if p.spmb <= 0:
        return -np.inf
    else: prior_spmb = p.spmb**-1
    if p.sdm <= 0:
        return -np.inf
    else: prior_sdm = p.sdm**-1

    prior_dm = 10**(2 * p.DM / 5)

    return np.log(prior_sb * prior_spml * prior_spmb * prior_sdm * prior_dm)

