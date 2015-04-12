import numpy as np
import matplotlib.pyplot as plt
mean = np.zeros(3)
index = np.arange(3)
for i in range(len(index)):
    s = np.random.poisson(3,10)
    mean[i] = np.mean(s)


a = np.mean(mean)
a_v = np.var(mean)
print a, a_v
print s
d = np.std(s)
print d
s[s<(d*3)] = 0
print s
c = s.sort()
print c
b = np.trim_zeros(c)
print b
print len(b)




        
  



        
