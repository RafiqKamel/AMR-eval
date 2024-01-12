import numpy as np

def vectorize_function(function, vector, otypes=None):
    vfun = np.vectorize(function, otypes=otypes)
    return vfun(vector)