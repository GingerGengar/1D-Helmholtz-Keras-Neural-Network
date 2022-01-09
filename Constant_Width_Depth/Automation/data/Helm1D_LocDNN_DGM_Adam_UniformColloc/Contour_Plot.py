import numpy as np
import matplotlib.pyplot as plt

#Reading from Files
data = np.genfromtxt("NN_Architecture_Error.txt", comments="variables")

xgrid = np.array([data[:,1],data[:,1]])
ygrid = np.array([data[:,0],data[:,0]])
val = 

#Plotting Pressure Contour
figure, axis = plt.subplots()
levels = np.linspace(-1.0,1.0,20)
CP1 = axis.contourf(xgrid, ygrid, val, levels,cmap="hot")
cbar = figure.colorbar(CP1)
axis.set_xlim(-6,6)
axis.set_ylim(-4,4)
axis.set_xlabel('Number of Hidden Layers')
axis.set_ylabel('Nodes on Each Hidden Layers')
plt.savefig('Error_Contour.png',dpi=1000)
