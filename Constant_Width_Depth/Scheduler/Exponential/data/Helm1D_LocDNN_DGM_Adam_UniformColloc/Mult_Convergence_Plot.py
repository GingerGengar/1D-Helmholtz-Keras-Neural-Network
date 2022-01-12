#Author: Hans C. Suganda
import numpy as np
import matplotlib.pyplot as plt
import glob

#Files to Open
filename_list = glob.glob('*_5.csv')

#Plotting
plt.figure()
for filename in filename_list:
    data = np.genfromtxt(filename, delimiter=',', comments=",loss,")
    plt.plot(data[:,0], data[:,1], label = filename)
plt.xlabel('Epochs')
plt.ylabel('loss-value')
plt.title('Exponential Training Convergence')
plt.legend()
plt.grid()

#Show Plot
plt.show()


