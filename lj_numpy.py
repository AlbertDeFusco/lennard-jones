import numpy as np

def lj(r):
    sr6 = (1./r)**6
    pot = 4.*(sr6*sr6 - sr6)
    return pot
    
    
def distances(cluster):
    diff = cluster[:, np.newaxis, :] - cluster[np.newaxis, :, :]
    mat = np.sqrt((diff*diff).sum(-1))
    return mat

    
def potential(cluster):
    d = distances(cluster)
    dtri = np.triu(d)
    energy = lj(dtri[dtri > 1e-6]).sum()
    return energy