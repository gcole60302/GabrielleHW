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
##############################################################
#Make 1D array with N=1024 points
I = np.arange(1,1025)
#Make t_i = i * delta t
delta_t = 3.7
T = np.zeros(1024)
for i in range(1023):
    T[i+1] = I[i]*delta_t
#Create sine wave, frequency = 0.001658
S = np.sin(((2.)*(np.pi)*(T))/(1024*delta_t))
#Plot sin curve
plt.figure()
plt.xlabel('T')
plt.ylabel('Sin(T)')
plt.title('Sine Wave vs. T')
plt.plot(T, S)
##############################################################
#FFT of the sine wave
FFT = scipy.fftpack.fft(S)
F1 = np.linspace(0.0, (2)*(np.pi)/(1024*delta_t), 1024)
Fi = ((T/delta_t) - (1024/2) + 1.)/((1024)*(delta_t))
#Plot fft of sine wave
plt.figure()
plt.xlabel('Frequency')
plt.ylabel('|FFT(Frequency)|')
plt.title('FFT of Sine Wave')
plt.plot(F1, np.abs(FFT))
#Plot FFT vs. Fi
plt.figure()
plt.xlabel('Frequency')
plt.ylabel('|FFT(Frequency)|')
plt.title('FFT vs Fi')
plt.plot(Fi, (1024/2)*np.abs(FFT))
##############################################################
#Form signal and noise arrays
X = np. linspace(0,10,1011)
N = np.random.normal(0,3, 1011)
S = np.sin(((2.)*(np.pi)*(X))/(5)) + np.cos(2*np.pi*X*.73) + np.sin(2*np.pi*X*3)
#Plot of just the signal
plt.figure()
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Pure Signal')
plt.plot(X, S)
#Plot of the noise plus the signal
plt.figure()
plt.xlabel('Time')
plt.ylabel('Total Signal')
plt.title('Signal Plus Mean Zero Gaussian Noise')
plt.plot(X, N+S)
#So there are 102 samples taken per second. So the sampling
#frequency is 102 Hz and therefore the highest frquency that
#may be observed in the sample is 51Hz. One tenth of the
#Nyquist frequency is 10.2 Hz, so we now cut the data, discarding
#all frequencies above this cutoff. 
##############################################################
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
#Define filter bans
fs = 102
lowcut = 1
highcut= 10.2
#Plot Butter filtered signal
Y = butter_bandpass_filter(N+S, lowcut, highcut, fs, order=6)
plt.figure()
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Signal filtered With a Butter Filter of 10.2 Hz')
plt.plot(X, Y)
##############################################################
#Plot autocorrelation function
R = np.zeros(1010)
for i in range(1010):
    R[i] = Y[i]*Y[i+1]
R = np.lib.pad(R, (0,1), 'constant')
plt.figure()
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Autocorrelation Function (1/102 sec lag)')
plt.plot(X, R)
##############################################################
#Plot LSD function
LSD = np.zeros(1011)
for i in range(1011):
    LSD[i] = np.abs(Y[i])/np.sqrt(102)
X1 = np. linspace(0,10,1011)
plt.figure()
plt.plot(X1, LSD*((1/102.)**2/(10)))
#Verification of RMS = integral of LSD
def rms(num):
    return sqrt(sum(n*n for n in num)/len(num))
print rms(Y)
print scipy.integrate.simps(LSD,X1)
#Plot of PSD
f1, Pxx_den1 = signal.welch(Y,102)
plt.figure()
plt.xlabel('Frequency')
plt.ylabel('Amplitude (dB)')
plt.title('Power Spectral Density')
plt.plot(f1, np.log10(Pxx_den1))
