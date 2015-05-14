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
NUM_EXP = 3 #However much time a person has to wait
NUM_SAMPLES = 200 #Ditto ^^

#Theta runs over angles -2pi to 2pi and the PDF is both defined in its orginal
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

##################################################
#LHE
#Constant term of LH function is dependent on # of samples in given experimental run
CONST= ((1.)/((4.)*(np.pi)))**(NUM_SAMPLES)

#This performs the successive multiplication to create the LH function for a given
#experimental run
def LHE(data, x):
    p = 1.
    for i in range(len(data)):
        p = p *(1. + (x)*(np.cos(data[i])))
    return p

#This is the range over which I'd like to plot the LH function
x = np.arange(0,1.,0.01)
LHE_VAL = np.zeros(len(x))
LH_P_VAL = np.zeros(NUM_EXP)

#This plots the LH function for a given experimental run over the range of possible
#P values defined by x(above). Also the max value of this function is printed out and
#stored, as the value suggested for P by that experimental run based on maximizing the
#LH function
for i in range(NUM_EXP):
    plt.figure()
    plt.axvline(0.5,0,1.3) #A vertical line marking the true value
    plt.title('Likelihood Function')
    plt.xlabel('Possible Values of P')
    plt.ylabel('Log Likelihood')
    plt.plot(x, np.log((CONST)*LHE(DATA[i],x)))
    for j in range(len(x)):
        LHE_VAL[j] = np.log((CONST)*LHE(DATA[i],x[j]))
    MAX = max(LHE_VAL)
    for k in range(len(x)):
        if np.log((CONST)*LHE(DATA[i],x[k])) == MAX:
            break
    LH_P_VAL[i] = x[k]
    print 'This is the LH estimation of the value of P', x[k]

#Variance of vales for P based on maximizing the likelihood function through all of the
#number of experiments
VAR_LH = np.var(LH_P_VAL)
print 'This is the variance of the LH', VAR_LH

###############################################
#KS Statistic
#Defines the cumulative distribution function based on the PDF over x ranges of -2pi to 2pi
#part of what we need to get the KS statistic
def CDF(x):
    return ((1.)/((4.)*(np.pi)))*((x)+(0.5)*(np.sin(x))+((2.)*(np.pi)))

#An empty array for possible KS statistics
DIFF = np.zeros(len(x))
MAX_DIFF = np.zeros(NUM_EXP)
#Calculates the empirical distribution functions (EDF) for all experimental runs and plots the corresponding
#step plot. Plot CDF function on top and find the max. differenc between the
#EDF and CDF to get the KS statistic, which is then printed out
for i in range(NUM_EXP):
    ecdf = ECDF(DATA[i])
    x = np.linspace(min(DATA[i]), max(DATA[i]))
    y = ecdf(x)
    plt.figure()
    plt.title('Empirical Distribution Function & CDF Function')
    plt.xlabel('Sample Values')
    plt.ylabel('Sum of Included Samples over Total Number of Samples')
    plt.plot(x,CDF(x))
    plt.step(x,y)
    for j in range(len(x)):
        DIFF[j] = np.abs(y[j] - CDF(x[j]))
    MAX_DIFF[i] = max(DIFF)
    print 'This is the KS statistic', max(DIFF)
    
VAR_KS = np.var(MAX_DIFF)
print 'This is the variance of the KS statistic', VAR_KS
    
###############################################
#Chisquared
#A useful defintion for all of this
def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

#We define the function CHI which sums over the chisquared statistic for all
#bins defined by an 11 bin histogram. We define the expected value by the intergral
#of the normalized pdf over the lowest to highest value found in a given bin.
def CHI(data, x):
    n, e = np.histogram(data, bins=11)
    d = np.digitize(data, e)  
    SUM = 0
    for j in range(12):
        F = np.zeros(len(d))
        for k in range(len(d)):
            if d[k]== j+1:
                F[k] = data[k]
        A = remove_values_from_list(F,0)
        Max = max(A)
        Min = min(A)
        SUM = SUM + (((n[i]) - ((((len(data))/(4.*np.pi))*((Max) + (x)*(np.sin(Max)) - (Min) - (x)*(np.sin(Min))))))**(2.))/(((len(data))/(4.*np.pi))*((Max) + (x)*(np.sin(Max)) - (Min) - (x)*(np.sin(Min))))
    return SUM

#Values over which Chi squared is plotted and minimized. Additionally some empty sets for the variances
x1 = np.arange(0.3,0.7, 0.07)
CHI_VAL = np.zeros(len(x1))
CHI_P_VAL = np.zeros(NUM_EXP)

#Creates plots and estimates the value of P. over a narrow range of possible values
#as the plot seems to be cyclic
for i in range(NUM_EXP):
    plt.figure()
    plt.title('Chisquared Statistic')
    plt.xlabel('Possiable P Values')
    plt.ylabel('Chi Squared')
    plt.axvline(0.5,0,40) #A vertical line marking the true value
    plt.plot(x1, CHI(DATA[i],x1))
    for j in range(len(x1)):
        CHI_VAL[j] = CHI(DATA[i],x1[j])
    MIN = min(CHI_VAL)
    for k in range(len(x1)):
        if CHI(DATA[i],x1[k]) == MIN:
            break
    CHI_P_VAL[i] = x1[k]
    print 'This is the Chisquared estimation of the value of P', x1[k]

#Variance of vales for P based on minimizing chisquared through all of the
#number of experiments
VAR_CHI = np.var(CHI_P_VAL)
print 'This is the variance of chisquared', VAR_CHI
############################################
#Seems like the power of Chisquared
