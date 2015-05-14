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

def CCI(coverage, Aflips, Bflips, Aheads, Bheads):
    P_B = np.float(Bheads)/Bflips
    P_A = np.float(Aheads)/Aflips
    Theta = np.float(P_B)/np.float(P_A)
    a = np.zeros(1)
    b = np.zeros(1)
    y = np.random.normal(Theta, 0.1, 1000)
    count, bins, ignored = plt.hist(y, 30, normed=True)
    #plt.plot(bins, 1/(0.1 * np.sqrt(2 * np.pi)) *np.exp( - (bins - Theta)**2 / (2 * 0.1**2)), linewidth=2, color='r')
    #plt.show()
    mean = Theta
    sigma = 0.1
    i = np.arange(-1000,0,0.01)
    j = np.arange(mean,1000,0.1)
    for k in range(len(i)):
        if quad(Interg, -1*np.inf, i[k] , args=(mean, sigma))[0] > 0.5 - coverage/2.:
            break
    a[0] = i[k]
    for k in range(len(j)):
        if quad(Interg, j[k], np.inf, args=(mean, sigma))[0] > coverage/2.:
            break
    b[0] = j[k]
    return a[0], b[0]

THETA = np.zeros(100)
NUM_AFlips = 100.*np.ones(100)
NUM_BFlips = 100.*np.ones(100)

x = np.arange(0.01,0.99,0.01)

PROB_A = .1 #Or any other prob
PROB_B = .4 #Or any other prob

NUM_AHeads = np.random.binomial(100, PROB_A, 100)
NUM_BHeads = np.random.binomial(1000, PROB_B, 100)

CI = np.zeros((100,2))
for i in range(100):
    THETA[i] = ((NUM_BHeads[i])/(100.))/((NUM_AHeads[i])/(100.))
    coverage = 0.68
    CI[i] = CCI(coverage, 100, 100, NUM_AHeads[i], NUM_BHeads[i])

print "Experimental Theta is:", THETA
print "True Theta is", PROB_B/PROB_A
print "True P_A is:", PROB_A
print "True P_B is:", PROB_B
print "Confidence Int. Are", CI
#####################   




