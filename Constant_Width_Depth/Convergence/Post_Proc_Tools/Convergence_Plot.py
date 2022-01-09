#Author: Hans C. Suganda
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np
import sys

#Command Line Input Name
solname = sys.argv[1]

#Reading from Files
history = np.genfromtxt(solname, delimiter=',', comments=",loss,")

#Plotting Results
plt.figure()
plt.plot(history[:,0], history[:,1])
plt.xlabel('Epoch')
plt.ylabel('loss-value')
plt.title('Training Convergence')
plt.grid()
plt.savefig('Train_Convergence.png', dpi = 1000)

#Show Plot
plt.show()

