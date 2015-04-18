import numpy as np
import matplotlib.pyplot as plt

#For Part A we generate the Poisson distribution
x = np.random.poisson(3, 10000)
plt.hist(x, bins=12, normed=True)
plt.ylabel('Frequency')
plt.xlabel('Measured/Obtained Value')
plt.title('Poisson Distr. Plot (Mean=3)')
plt.show()

#For Parts B, C we create an array 101x50 of values
#randomly choosen from our defined distribution
b = np.zeros((101,50))
for i in range(101):
    a=np.random.choice(x, 50)
    b[i]= a

#For Part D we create an array of mean values for each
#'simulated' data set and plot in a histogram
d= np.zeros(101)
for i in range(101):
    d[i]= np.mean(b[i])

plt.hist(d, bins=10, normed=True)
plt.ylabel('Frequency')
plt.xlabel('Mean of Simulated Data')
plt.title('Sample Mean Values')
plt.show()
