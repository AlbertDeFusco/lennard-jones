import numpy as np
import numba

@numba.jit(nopython=True, nogil=True)
def lj(r):
    sr3 = (1./r)**3
    pot = 4.*(sr3*sr3 - sr3)
            
    return pot
    
@numba.jit(nopython=True, parallel=True)
def potential(cluster):
    pot = 0.0
    n = cluster.shape[0]
    for i in range(n-1):
        for j in numba.prange(i+1, n):
            dx = cluster[j,0] - cluster[i,0]
            dy = cluster[j,1] - cluster[i,1]
            dz = cluster[j,2] - cluster[i,2]
            
            d2 = dx*dx + dy*dy + dz*dz
            pot += lj(d2)
            
    return pot