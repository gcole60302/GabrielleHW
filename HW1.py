#some imports
import numpy as np
import matplotlib.pyplot as plt

#create sample size N = 1,000,000 with distribution mean of 3
s= np.random.poisson(3,1000)

#create histogram
count, bins, ignored = plt.hist(s, bins=20, range=(0,20), normed=True)

#some plot lables 
plt.xlabel("Could be anything")
plt.ylabel("")
plt.title("Histogram of Distribution")


#some data on s
print np.max(s), np.min(s)
print np.mean(s)

#print plot
plt.show()


        
