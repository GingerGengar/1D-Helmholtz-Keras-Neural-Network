#Author: Hans C. Suganda
import matplotlib.pyplot as plt
import numpy as np

#Reading from Files
data = np.genfromtxt("breadth20.dat")

#Parsing the titles
data = data[1::,:]

#Error
aveError = np.mean(data[:,3])
maxError = max(data[:,3])
print(maxError, aveError)

#Plotting Solutions Plot
plt.figure(1)
plt.plot(data[:,0], data[:,1], '-r', label='NN')
plt.plot(data[:,0], data[:,2], '-k', label='True')
plt.xlabel('x-value')
plt.ylabel('y-value')
plt.title('Solution Comparison')
plt.legend()
plt.grid()
plt.savefig('breadthsol20.png', dpi=500)

#Plotting Error
plt.figure(2)
plt.plot(data[:,0], data[:,3], '-r', label='Error')
plt.xlabel('x-value')
plt.ylabel('Error')
plt.title('Error Vs Length')
plt.legend()
plt.grid()
plt.savefig('breadtherr20.png', dpi=500)

plt.show()
