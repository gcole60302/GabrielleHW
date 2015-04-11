import numpy as np
import matplotlib.pyplot as plt
s= np.random.poisson(5,100000)
count, bins, ignored = plt.hist(s, 19, normed=True)
plt.xlabel("")
plt.ylabel("")
plt.title("Poisson distribution")
plt.show()

        
