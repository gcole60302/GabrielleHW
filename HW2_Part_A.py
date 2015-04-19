import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.misc
import math
from sympy import *
import scipy.optimize as opt

#For Part A we generate the Poisson distribution
x = np.random.poisson(3, 10000)
plt.hist(x, bins=12, normed=True)
plt.ylabel('Frequency')
plt.xlabel('Measured/Obtained Value')
plt.title('Poisson Distr. Plot (Mean=3)')
plt.show()

#For Parts B, C we create an array 101x50 of values
#randomly choosen from our defined distribution
#b[0] = 'data', b[1-100] = 'simulations'
b = np.zeros((101,50))
for i in range(101):
    a=np.random.choice(x, 50)
    b[i]= a

#For Part D we create an array of mean values for each
#'simulated' data set and plot in a histogram
d= np.zeros(101)
for i in range(101):
    d[i]= np.mean(b[i])

plt.hist(d, bins=10, normed=True)
plt.ylabel('Frequency')
plt.xlabel('Mean of Simulated Data')
plt.title('Sample Mean Values')
plt.show()

#For Part E we let mu run over ranges -10 to 10. We plot the
#proablity distribution as a function of this mu. We use our
#mean values from our simulations in Part D for each sucessive
#plot. Dashed black line is 'data'. 7 is a placeholder to clarify
#plots created. 101 could be used just as well.
mu = np.arange(0,7,0.001)
for i in range(5):
    q = ((mu**(d[i+1]))*((np.e)**(-mu)))/(scipy.misc.factorial(d[i+1]))
    plt.plot(mu, q)
    plt.ylabel('Probability (log scaled)')
    plt.xlabel('Mu Values')
    plt.title('Likelihood Functions for Simulations & Data')
    plt.yscale('log')
plt.plot(mu, ((mu**(d[0]))*((np.e)**(-mu)))/(scipy.misc.factorial(d[0])), color='black',ls='--',lw=5)
plt.show()

#For Part F we max the likelihood functions for each simulation and
#the data. Then we saved in in an array called Poss_Mu values and
#printed out a simple variance from these max values
g= np.arange(0,10,0.001)
Poss_Mu = np.zeros(101)
for i in range(101):
    f = ((g**(d[i]))*((np.e)**(-g)))/(scipy.misc.factorial(d[i]))
    max_y = max(f)
    Poss_Mu[i] = g[f.argmax()]
print Poss_Mu
print np.var(Poss_Mu)

#Produces a plot of ln(L(mu, x)) as function of x. Also produces line
#that divides one sigma variance point. The intercetion of the curves
#defines the one sigma error bars.
const1 = (-1.495922603)*np.ones(499)
tau= np.arange(0.01,5,.01)
L = np.log(((3.)**(tau))*((np.e)**(-3.))/(scipy.misc.factorial(tau)))
plt.plot(tau, L)
plt.plot(tau, const1)
plt.show()

#Outputs a percentage of Poss_Mu that fall into the one sigma error bars
upper = 3. + np.var(Poss_Mu)
lower = 3. - np.var(Poss_Mu)
for i in range(len(Poss_Mu)):
    if Poss_Mu[i] > upper:
        Poss_Mu[i]=0
    if Poss_Mu[i] < lower:
        Poss_Mu[i]=0
     
Mu2= np.asarray(Poss_Mu)
Sum = (Mu2 >0).sum(dtype=float)
print (100)*(Sum/101.)
