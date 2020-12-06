# 2020 update

Attempting to bring some sanity to this and see if I can still make it work. Tread carefully, there is some seriously terrible code here

## Installing

This was a pain in the arse. Eventually I used Anaconda which seemed to be the only decent way to get `gala` to work on Windows. Need to make sure env variables are up to date with the python executable in the anaconda directory, not the Python directory, and also add one for anaconda3\Library\bin (https://github.com/numpy/numpy/issues/14770), otherwise `numpy` poos itself on import. Idk why. Conda install `numpy`, `astropy`, `gala` in order seemed to work then.

If `numpy` is installed already by `pip`, it can screw with trying to install it again with Anaconda, so make sure no trace remains.

---

# MWMass

My MPhys project!!!

Update 3/1/19: uploaded mega_orbit_fit.py - the mcmc fitting script. Input data is sgr_input_data.npy and vc_rh_arrays_log.npy

actual_scripts/SUPERAUTOPHOTOMETRY (it's exciting ok) is the main aperture photometry notebook, including the star finding code. Will update soon with more comments.

megasheet is a csv table of (among other things):
CSS IDs
Gaia IDs
Co-ordinates from Gaia
Proper motions and parallaxes from Gaia, with errors
Periods, V-band extinctions and distances (for comparison) from Drake
Will update with other things as they come, such as metallicities, my calculated magnitudes and distances...
My code refers extensively to this sheet, as well as to the Spitzer data itself

Also in the python_expts folder is all of the messing around with Python I've done in the last couple of weeks including photutils and astropy things
