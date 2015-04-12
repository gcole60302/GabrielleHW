import numpy as np
import matplotlib.pyplot as plt
mean = np.zeros(10)
index = np.arange(10)
for i in range(len(index)):
    s = np.random.poisson(3,1000)
    mean[i] = np.mean(s)

print mean
a = np.mean(mean)
a_v = np.variance(mean)
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

print mean
b = np.mean(mean)
b_v = np.variance(mean)
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

print mean
c = np.mean(mean)
c_v = np.variance(mean)
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

print mean
d = np.mean(mean)
d_v = np.variance(mean)
print d, d_v


plt.figure()
plt.title('10000 Times')
plt.hist(mean, normed=True)
plt.show()
    



        



        
  



        
