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
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
###################################################
#DATA created
#Defines the number of experiments done and the number of samples drawn in
#each experiment
NUM_EXP = 2
NUM_SAMPLES = 10

#Theta runs over angles -2pi to 2pi and the PDF is both defined in it's orginal
#form and in its normalized form over the range of angles -2pi to 2pi
U = np.arange(0,1,0.01)
X = np.arange(-2.*np.pi, 2.*np.pi, 0.01)
DATA = np.zeros((NUM_EXP, NUM_SAMPLES))

def PDF_ORIG(x):
    return (1. + (0.5)*(np.cos(x)))

def PDF_NORMED(x):
    return (1./((4.)*(np.pi)))*(1. + (0.5)*(np.cos(x))) 

#Random data is created using inverse transform sampling from the normalized
#pdf over angles -2pi to 2pi.
for k in range(NUM_EXP):
    for i in range(NUM_SAMPLES):
        u = np.random.choice(U, 1)[0]
        for j in range(len(X)):
            if scipy.integrate.quad(PDF_NORMED, -2.*np.pi, X[j])[0] >= u: 
                break

        DATA[k,i] = X[j]
#######################################################

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]
########################
def CHI(data):
    n, e = np.histogram(data, bins=5)
    d = np.digitize(data, e)
    print d
    SUM = 0
    for j in range(6):
        F = np.zeros(len(d))
        for k in range(len(d)):
            if d[k]== j+1:
                F[k] = data[k]
        A=remove_values_from_list(F,0)
        print A
        Max = max(A)
        print Max
        Min = min(A)
        print Min
