import dill as pickle
import scipy.interpolate as interpolate
import numpy as np

interpolation = interpolate.interp1d(np.arange(0,10), np.arange(0,10))
with open("test_interp", "wb") as dill_file:
     pickle.dump(interpolation, dill_file)
with open("test_interp", "rb") as dill_file:
     interpolation = pickle.load(dill_file)