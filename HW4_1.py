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

###############################################
#Frequentist Approach
def THETA(Aflips, Bflips, Aheads, Bheads):
    E_ProbA = np.float(Aheads)/Aflips
    E_ProbB = np.float(Bheads)/Bflips

    E_Theta = np.float(E_ProbB)/np.float(E_ProbA)

    return E_Theta

q = np.zeros(100)
p = np.zeros(100)
u = np.zeros(100)
def CI(coverage, T_ProbA, T_ProbB):
    x = np.arange(0.01,0.99,0.01)
    y = np.arange(100,101)
    
    T_ProbA = T_ProbA #np.random.choice(x,1)[0] #true prob of heads for coin a
    T_ProbB = T_ProbB #np.random.choice(x,1)[0] #true prob of heads for coin b

    T_Theta = T_ProbB/T_ProbA       #true value of theta

    Num_FlipsA = np.random.choice(y,100)    #array of 100 random numbers of flips in a given experiment (of flipping coin a)
    Num_FlipsB = np.random.choice(y,100)    #array of 100 random numbers of flips in a given experiment (of flipping coin b)

    for i in range(100):
        E_Num_HeadsA= np.random.binomial(Num_FlipsA[i], T_ProbA, 1) #flip coin A 25 times, 100 times over.
        E_Num_HeadsB= np.random.binomial(Num_FlipsB[i], T_ProbB, 1) #flip coin B 25 times, 100 times over.

        p[i] = np.float(E_Num_HeadsA[0])/Num_FlipsA[i]
        u[i] = np.float(E_Num_HeadsB[0])/Num_FlipsB[i]
        
        q[i] = THETA(np.float(Num_FlipsA[i]), np.float(Num_FlipsB[i]), np.float(E_Num_HeadsA[0]), np.float(E_Num_HeadsB[0]))
    plt.hist(q, 16, normed=1)
    plt.show()
    Nsigma = np.sqrt(2) * erfinv(coverage)
    mu = q.mean()
    sigma_mu = (1.0) * q.size **(-0.5)
    print "True Theta is:", T_Theta
    print "True Prob Coin A is Heads", T_ProbA
    print "True Prob Coin B is Heads", T_ProbB

    print "Exp Theta is:", mu
    print "Prob Coin A is Heads", np.mean(p)
    print "Prob Coin B is Heads", np.mean(u)
    

    print "Lower Bound CI", mu - Nsigma * sigma_mu
    print "Upper Bound CI", mu + Nsigma * sigma_mu
    if T_Theta > mu - Nsigma * sigma_mu and T_Theta < mu + Nsigma * sigma_mu:
        print "CI Covers"
    else:
        print "CI Doesn't Cover"
    return 
#As the parameter theta increases the coverage decreases, that is as the
#prob of getting heads for B increases and the proablity for getting heads
#for A decreases.
##################################################
#Bayesian Approach 

def CI(coverage, T_ProbA, T_ProbB):
    x = np.arange(0.01,0.99,0.01)
    y = np.arange(100,101)
    
    T_ProbA = T_ProbA #np.random.choice(x,1)[0] #true prob of heads for coin a
    T_ProbB = T_ProbB #np.random.choice(x,1)[0] #true prob of heads for coin b

    T_Theta = T_ProbB/T_ProbA       #true value of theta

    Num_FlipsA = np.random.choice(y,100)    #array of 100 random numbers of flips in a given experiment (of flipping coin a)
    Num_FlipsB = np.random.choice(y,100)    #array of 100 random numbers of flips in a given experiment (of flipping coin b)

    for i in range(100):
        E_Num_HeadsA= np.random.binomial(Num_FlipsA[i], T_ProbA, 1) #flip coin A 25 times, 100 times over.
        E_Num_HeadsB= np.random.binomial(Num_FlipsB[i], T_ProbB, 1) #flip coin B 25 times, 100 times over.

        p[i] = np.float(E_Num_HeadsA[0])/Num_FlipsA[i]
        u[i] = np.float(E_Num_HeadsB[0])/Num_FlipsB[i]
        
        q[i] = THETA(np.float(Num_FlipsA[i]), np.float(Num_FlipsB[i]), np.float(E_Num_HeadsA[0]), np.float(E_Num_HeadsB[0]))
    plt.hist(q, 16, normed=1)
    plt.show()
    Nsigma = np.sqrt(2) * erfinv(coverage)
    mu = q.mean()
    sigma_mu = (1.0) * q.size ** -0.5
    return mu - Nsigma * sigma_mu, mu + Nsigma * sigma_mu
