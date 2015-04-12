import numpy as np
import matplotlib.pyplot as plt
mean = np.zeros(3)
index = np.arange(3)
for i in range(len(index)):
    s = np.random.poisson(3,1000) # so 1000 is our N, the larger it gets the smaller our final value becomes
    mean[i] = np.mean(s)


a = np.mean(mean)
a_v = np.var(mean)
d = np.std(s)

    
print a, a_v
print s

print d
s[s<(d*3)] = 0
print s
s.sort()
print s
print len(s)
b = np.trim_zeros(s)
print b
print len(b)

print len(b)/float(len(s))
#as N gets larger this value the measure of non-G gets smaller

        
  



        
