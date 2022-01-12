#Author: Hans C. Suganda
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np
import sys

#Command Line Input Name
solname1 = sys.argv[1]
solname2 = sys.argv[2]

#Reading from Files
history1 = np.genfromtxt(solname1, delimiter=',', comments=",loss,")
history2 = np.genfromtxt(solname2, delimiter=',', comments=",loss,")

#Plotting Results
plt.figure()
plt.plot(history1[:,0], history1[:,1], label=solname1)
plt.plot(history2[:,0], history2[:,1], label=solname2)
plt.xlabel('Epoch')
plt.ylabel('loss-value')
plt.title('Training Convergence')
plt.grid()
plt.savefig('Train_Convergence.png', dpi = 1000)
plt.legend()

#Show Plot
plt.show()

