#Author: Hans C. Suganda
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np
import convergence
import sys

#Command Line Solution Input Name
solname = sys.argv[1]

#Command Line Time History Name
histname = sys.argv[2]

#Command Line Width Breadth of NN
width = int(sys.argv[3])
depth = int(sys.argv[4])

#Reading from Files
data = np.genfromtxt(solname, comments="variables")
history = np.genfromtxt(histname, delimiter=',', comments=",loss,")

#Convergence
Converge = convergence.run(history[:,1])

#Average Error
aveError = np.mean(data[:,3])

#Maximum Error
maxError = max(data[:,3])

#L^2 Error
L2 = LA.norm(data[:,3])/(np.sqrt(data[:,3].size))

print(width, depth, aveError, maxError, L2, Converge)

