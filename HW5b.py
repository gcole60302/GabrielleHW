import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.misc
import math
from sympy import *
from scipy import integrate
import scipy.optimize as opt
from scipy.optimize import curve_fit
from matplotlib.ticker import NullFormatter, MaxNLocator
from numpy import linspace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
import scipy.misc
import math
import matplotlib.image as mpimg
from numpy import linspace
from sympy import *
import scipy.optimize as opt
import scipy.interpolate
import random
from numpy.random import uniform
import scipy.stats as st
from sympy.stats import *
plt.ion()
from scipy import stats
#########################################
def f(x):
     return x**3 + x**2

print scipy.misc.derivative(f, 1.0, dx=1e-6)
