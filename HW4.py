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
###############################################

def THETA_CONF(coverage, Atries, Btries, Aheads, Bheads):
    ExpProbHeadA = Aheads/np.float32(Atries)
    ExpProbHeadB = Bheads/np.float32(Btries)
    Theta = ExpProbHeadB/ExpProbHeadA
    a = 1.0*np.array(Theta)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+coverage)/2., n-1)
    count, bins, ignored = plt.hist(Theta, 16, normed=True, facecolor='red')
    mu, sigma = norm.fit(Theta)
    plt.axvline(m-h)
    plt.axvline(m+h)
    plt.plot(bins, 1/(sigma * np.sqrt(2.* np.pi)) * np.exp(-(bins - mu)**2./(2. * sigma**2)), linewidth=2, color='b')
    plt.show()
    return m, m-h, m+h  
