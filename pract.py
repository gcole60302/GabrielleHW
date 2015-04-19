import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.misc
import math
from sympy import *
import scipy.optimize as opt

tau= np.arange(0,10,0.001)
L = log(((3)**(tau))*((np.e)**(-3))/(scipy.misc.factorial(tau))
plt.plot(tau,L)
plt.show()
