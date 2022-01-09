#Author: Hans C. Suganda
import numpy as np

"""
ind represents the independent variable, which should be a numpy array. To test for convergence, the function below takes average gradient at the earlier stages of the given array. The function does the same but for later stages of the given array, then if the gradient at the later stages of the array is much less than the initial gradient, then the given array is assumed to have indicated "convergence" of the numerical scheme
"""

def run(ind):
    cutoff = 0.03 #Allowable percentage difference of the gradients
    percfront = 0.2 #percentage from the front for the index
    percback = 0.3 #percentage from the back for the index
    elsize = ind.size #Necessary for indexing
    
    #Initial Gradient
    gradinit = ind[int(elsize*percfront)] - ind[0]
    
    #Final Gradient
    gradfin = ind[elsize-1] - ind[int(elsize*(1.0-percback))]

    #Convergence Criterion
    result = False
    if(abs(gradfin)<(cutoff*abs(gradinit))):
        result = True
    
    return result
