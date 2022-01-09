#Author: Hans C. Suganda
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np
import sys

#Command Line Input Name
solname = sys.argv[1]

#Command Line Width Breadth of NN
width = int(sys.argv[2])
depth = int(sys.argv[3])

#Reading from Files
data = np.genfromtxt(solname, comments="variables")

#Average Error
aveError = np.mean(data[:,3])

#Maximum Error
maxError = max(data[:,3])

#L^2 Error
L2 = LA.norm(data[:,3])

print(width, depth, aveError, maxError, L2)

