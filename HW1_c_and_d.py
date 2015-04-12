import numpy as np
import matplotlib.pyplot as plt
mean = np.zeros(10)
index = np.arange(10)
for i in range(len(index)):
    s = np.random.poisson(3,1000)
    mean[i] = np.mean(s)


a = np.mean(mean)
a_v = np.var(mean)
print a, a_v


plt.figure()
plt.title('10 Times')
plt.hist(mean, normed=True)
plt.show()
    
#############

import numpy as np
import matplotlib.pyplot as plt
mean = np.zeros(100)
index = np.arange(100)
for i in range(len(index)):
    s = np.random.poisson(3,1000)
    mean[i] = np.mean(s)


b = np.mean(mean)
b_v = np.var(mean)
print b, b_v


plt.figure()
plt.title('100 Times')
plt.hist(mean, normed=True)
plt.show()

#############

import numpy as np
import matplotlib.pyplot as plt
mean = np.zeros(1000)
index = np.arange(1000)
for i in range(len(index)):
    s = np.random.poisson(3,1000)
    mean[i] = np.mean(s)


c = np.mean(mean)
c_v = np.var(mean)
print c, c_v

plt.figure()
plt.title('1000 Times')
plt.hist(mean, normed=True)
plt.show()
    
#############

import numpy as np
import matplotlib.pyplot as plt
mean = np.zeros(10000)
index = np.arange(10000)
for i in range(len(index)):
    s = np.random.poisson(3,1000)
    mean[i] = np.mean(s)


d = np.mean(mean)
d_v = np.var(mean)
print d, d_v


plt.figure()
plt.title('10000 Times')
plt.hist(mean, normed=True)
plt.show()


#############

import numpy as np
import matplotlib.pyplot as plt
mean = np.zeros(100000)
index = np.arange(100000)
for i in range(len(index)):
    s = np.random.poisson(3,1000)
    mean[i] = np.mean(s)


e = np.mean(mean)
e_v = np.var(mean)
print e, e_v


plt.figure()
plt.title('100000 Times')
plt.hist(mean, normed=True)
plt.show()    
#############


import numpy as np
import matplotlib.pyplot as plt
mean = np.zeros(1000000)
index = np.arange(1000000)
for i in range(len(index)):
    s = np.random.poisson(3,1000)
    mean[i] = np.mean(s)

f = np.mean(mean)
f_v = np.var(mean)
print f, f_v


plt.figure()
plt.title('1000000 Times')
plt.hist(mean, normed=True)
plt.show()    
#############

TotalMean = [a, b, c, d, e, f]
TotalVar = [a_v, b_v, c_v, d_v, e_v, f_v]
Axis = np.arange(0,6)
plt.plot(Axis, TotalMean, marker='o', linestyle='None')
plt.show()
plt.plot(Axis, TotalVar, marker='o', linestyle='None')
plt.show()


        
  



        
