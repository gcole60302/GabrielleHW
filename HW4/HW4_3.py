import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import scipy.misc
import math
from sympy import *
import scipy.optimize as opt
from scipy import integrate
import scipy.optimize as opt
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
import matplotlib.mlab as mlab
from scipy.stats import norm
from scipy.special import erfinv
from scipy.integrate import quad

###############################################
def Interg(x, mean, sigma):
    return ((1)/(np.sqrt((2.0)*(np.pi)*(sigma)**2)))*(np.exp(-((x)-(mean))**2/((2.)*(sigma)**2)))

def B_CR(mu_t, D, sigma, coverage):
    Nsigma = np.sqrt(2) * erfinv(coverage)
    mu = D.mean()
    sigma_mu = (1/mu_t)sigma * D.size ** -0.5
    return mu - Nsigma * sigma_mu, mu + Nsigma * sigma_mu

THETA = np.zeros(100)
NUM_AFlips = 100.*np.ones(100)
NUM_BFlips = 100.*np.ones(100)

x = np.arange(0.01,0.99,0.01)

PROB_A = .1 #Or any other prob
PROB_B = .4 #Or any other prob
T_THETA = PROB_B/PROB_A

NUM_AHeads = np.random.binomial(100, PROB_A, 100)
NUM_BHeads = np.random.binomial(1000, PROB_B, 100)

BCR = np.zeros((100,2))
for i in range(100):
    THETA[i] = ((NUM_BHeads[i])/(100.))/((NUM_AHeads[i])/(100.))
    sigma = 1.0
    coverage =0.76

BCR = B_CR(T_THETA, THETA, sigma, coverage)
print BCR
