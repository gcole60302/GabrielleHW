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

############################################
#Line y=1.5x+3 is generated
#Generates data, an array of 100 experiments each of 15 data points.
#Each of the 15 are in a sigma 0.6 gaussian distribution about
#the actual y value of the function (the line) at that point

Kx = np.arange(0,20,1)
Ky = 1.5*Kx+3
data = np.zeros((100,15))
for i in range(100):
    for j in range(15):
        data[i,j:j+1] = np.random.normal(Ky[j],0.6,1)    
#Plot one data set, number 5, and its +/- one sigma bars
#Plus plot the actual line Ky/Kx
Kx1 = np.arange(0,15,1)
plt.figure()
plt.ylim([-1,32])
plt.xlim([-1,20])
plt.xlabel('Known X Values')
plt.ylabel('')
plt.title('Experiment Number 4, +/- One-Sigma Error Bars & Established Function')
plt.scatter(Kx1, data[4])
plt.errorbar(Kx1, data[4], yerr=0.6, linestyle="None")
plt.plot(Kx,Ky)

##########################################
#Linear least squares fit for all 100 experiments, a histogram of slopes
#and  intercepts. The variance for both the slope data and intercpet data
#show that the estimators for the slope adn intercept are unbiased 
LLS = np.zeros((100,2))
for i in range(100):
    M= np.vstack([Kx1, np.ones(len(data[i]))]).T
    LLS[i] = np.linalg.lstsq(M,data[i])[0]

LLS_slope = np.zeros((100,1))
for i in range(100):
    LLS_slope[i] = LLS[i,0]

LLS_inter = np.zeros((100,1))
for i in range(100):
    LLS_inter[i] = LLS[i,1]   

print np.var(LLS_slope)
print np.var(LLS_inter)
plt.figure()
plt.xlabel('Slope Value')
plt.ylabel('Frequency')
plt.title('Distribution Of Slopes')
plt.hist(LLS_slope, 10)
plt.show()
plt.figure()
plt.xlabel('Intercept Value')
plt.ylabel('Frequency')
plt.title('Distribution of Intercepts')
plt.hist(LLS_inter, 10)


#######################################
#Chi squared values should follow a chi squared distribution
#characteristic of one degree of freedom, since each value is sampled
#from a guassian distribution. 
ChiSlope=((LLS_slope-1.5)**2.)/1.5
plt.figure()
plt.xlabel('Chi Squared for Slope Values')
plt.ylabel('Frequency')
plt.title('Distriution of Chi Squared for Slope')
plt.hist(ChiSlope, 10)
ChiInter=((LLS_inter-3)**2.)/3
plt.figure()
plt.xlabel('Chi Squared for Intercept Values')
plt.ylabel('Frequency')
plt.title('Distriution of Chi Squared for Intercept')
plt.hist(ChiInter, 10)
ReducedChi=np.zeros((100))
for i in range(100):
    ReducedChi[i]= (1./100.)*((LLS_slope[i]-1.5)**2.)/np.var(LLS_slope)
plt.figure()
plt.xlabel('Reduced Chi Squared Vaules')
plt.ylabel('Frequency')
plt.title('Distriution of Reduced Chi Squared')
plt.hist(ReducedChi, 10)


#####################################
#PART 3
Samples = np.zeros((100,1000))
for i in range(100):
    Samples[i] = np.random.exponential(20, 1000)
plt.figure()
plt.xlabel('')
plt.ylabel('')
plt.title('One Data Sample With Error Bars')
y, bin_edge = np.histogram(Samples[4],bins=20)
bin_centers = 0.5*(bin_edge[1:] + bin_edges[:-1])
plt.errorbar(bin_centers,y, yerr = y**0.5, drawstyle='steps-mid')
plt.show()

########################################
#PART 3

