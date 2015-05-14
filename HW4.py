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

def THETA_CONF_INTV(coverage, Atries, Btries, Aheads, Bheads):
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

###############################################

x = np.arange(0.01,.99,0.01)
y = np.arange(100,1000)

A_Tries = np.zeros((50,100))
B_Tries = np.zeros((50,100))
A_Heads = np.zeros((50,100))
B_Heads = np.zeros((50,100))
Prob_H_A = np.zeros((50,100))
Prob_H_B = np.zeros((50,100))
Theta = np.zeros((50,100))
Data = np.zeros((50,3))
def THETA(coverage):
    for i in range(50):
        for j in range(100):
            a_tries = np.random.choice(y,100)
            b_tries = np.random.choice(y,100)

            prob_heads_a = np.random.choice(x,100)
            prob_heads_b = np.random.choice(x,100)

            Prob_H_A[i] = prob_heads_a
            Prob_H_B[i] = prob_heads_b
        
            A_Tries[i] = a_tries
            B_Tries[i] = b_tries
        
        A_Heads[i] = np.ceil(Prob_H_A[i] * A_Tries[i])
        B_Heads[i] = np.ceil(Prob_H_B[i] * B_Tries[i])
    
        Theta[i] = Prob_H_B[i]/Prob_H_A[i]
    for k in range(50):
        a = Theta[k]
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t._ppf((1+coverage)/2., n-1)
        #count, bins, ignored = plt.hist(Theta[k], 16, normed=True, facecolor='red')
        #mu, sigma = norm.fit(Theta[k])
        #plt.figure()
        #plt.axvline(m-h)
        #plt.axvline(m+h)
        #plt.plot(bins, 1/(sigma * np.sqrt(2.* np.pi)) * np.exp(-(bins - mu)**2./(2. * sigma**2)), linewidth=2, color='b')
        #plt.show()
        Data[k,0] = m
        Data[k,1] = m-h
        Data[k,2] = m+h
    if Data[i,]
    return  
##############################################






