from __future__ import division
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
import scipy.fftpack
from scipy import signal
############################################################
X = np. linspace(0,10,1011)
N = np.random.normal(0,3, 1011)
S = np.sin(((2.)*(np.pi)*(X))/(5)) + np.cos(2*np.pi*X*.73) + np.sin(2*np.pi*X*3)
plt.figure()
plt.xlabel('')
plt.ylabel('')
plt.title('Pure Signal')
plt.plot(X, S)
plt.figure()
plt.xlabel('')
plt.ylabel('')
plt.title('Signal Plus Mean Zero Gaussian Noise')
plt.plot(X, N+S)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

fs = 102
lowcut = 1
highcut= 10.2
Y = butter_bandpass_filter(N+S, lowcut, highcut, fs, order=6)
plt.figure()
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Signal filtered With a Butter Filter of 10.2 Hz')
plt.plot(X, Y)

R = np.zeros(1010)
for i in range(1010):
    R[i] = Y[i]*Y[i+1]
R = np.lib.pad(R, (0,1), 'constant')
plt.figure()
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Autocorrelation Function (1/102 sec)')
plt.plot(X, R)

f1, Pxx_den1 = signal.welch(Y,102)
plt.figure()
plt.xlabel('')
plt.ylabel('')
plt.yscale('log')
plt.title('Power Spectral Density')
plt.plot(f1, Pxx_den1)

plt.figure()
plt.xlabel('')
plt.ylabel('')
plt.title('Linear Spectral Density')

plt.plot(f1, np.sqrt(Pxx_den1))

LSD = np.zeros(1011)
for i in range(1011):
    LSD[i] = np.abs(Y[i])/np.sqrt(102)
X1 = np. linspace(0,10,1011)
plt.figure()
plt.plot(X1, LSD*((1/102.)**2/(10)))

plt.figure()
plt.plot(X1,LSD*((1/102.)**2/(10))/(50))
print scipy.integrate.simps(LSD,X1)

def rms(num):
    return sqrt(sum(n*n for n in num)/len(num))

print rms(Y)

