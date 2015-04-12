#some imports
import numpy as np
import matplotlib.pyplot as plt
#3 is an arbitrary number, any number of plots can be generated
mean = np.zeros(3)
index = np.arange(3)
for i in range(len(index)):
    s = np.random.poisson(3,1000)
    mean[i] = np.mean(s)
    print np.mean(s)
    plt.figure(i)
    plt.hist(s, bins=20, range=(0,20), normed=True)
    plt.xlabel("Could be anything")
    plt.ylabel("'mean[i]'")
    plt.title("Histogram of Distribution")
    plt.show(i)
    
   



        
