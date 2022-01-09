#Author: Hans C. Suganda
import matplotlib.pyplot as plt
import numpy as np

#Reading from Files
data = np.genfromtxt("breadth.txt")

#Generate x-axis
x = np.linspace(100,20,5)

#Plotting Mean Error
plt.figure(1)
plt.semilogy(x, data[:,1], '-r', label='Average')
plt.xlabel('Hidden Layer Breadth')
plt.ylabel('Error')
plt.title('Average Error')
plt.legend()
plt.grid()
plt.savefig('breadthMean.png', dpi=500)

#Plotting Maximum Error
plt.figure(2)
plt.semilogy(x, data[:,0],'-b', label='Maximum')
plt.xlabel('Hidden Layer Breadth')
plt.ylabel('Error')
plt.title('Maximum Error')
plt.legend()
plt.grid()
plt.savefig('breadthMax.png', dpi=500)

plt.show()
