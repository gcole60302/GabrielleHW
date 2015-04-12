import numpy as np
import matplotlib.pyplot as plt
#some imports

s= np.random.poisson(3,1000000)
#create sample size N = 1,000,000 with distribution mean of 3

count, bins, ignored = plt.hist(s, bins=100, range=(), normed=True)

plt.xlabel("")
plt.ylabel("")
plt.title("Histogram of Distribution")
#some plot lables 
plt.show()

        
