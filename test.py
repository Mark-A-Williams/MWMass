# A test file to see what's working. Most of it isn't.

import sys

# print(sys.executable)
# for p in sys.path:
#     print(p)

import numpy as np

print(np.random.rand())

import astropy as astropy

print(astropy.astronomical_constants.get())

import gala as gala

print(gala.coordinates.galactocentric.get_galactocentric2019())
