import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
import scipy.misc
import math
from sympy import *
import scipy.optimize as opt
from scipy import stats
plt.ion()

#####################################
x=11.93394592

MaxR = np.int8(np.ceil(x))
if MaxR %2 == 0:
    MaxR = MaxR + 1
else:
    MaxR = MaxR
