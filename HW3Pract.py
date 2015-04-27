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

#####################################

vects = np.linspace(0, 5,21)
x,y = np.meshgrid(vects, vects)
DistR = np.zeros((21, 21))
for i in range(21):
    for j in range(21):
        DistR[i,j] = np.sqrt((x[10,10] - x[i,j])**2 +(y[10,10] - y[i,j])**2)

