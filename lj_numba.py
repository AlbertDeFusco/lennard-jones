
import numba
# deocorate the pure Python functions

@numba.jit(nopython=True, nogil=True)
def lj(r):
    sr6 = (1./r)**6
    pot = 4.*(sr6*sr6 - sr6)
    return pot


@numba.jit(nopython=True, nogil=True)
def distance(atom1, atom2):
    dx = atom2[0] - atom1[0]
    dy = atom2[1] - atom1[1]
    dz = atom2[2] - atom1[2]

    r = (dx*dx + dy*dy + dz*dz)**0.5
    return r


@numba.jit(nopython=True, nogil=True)
def potential(cluster):
    energy = 0.0
    for i in range(len(cluster)-1):
        for j in range(i+1,len(cluster)):
            r = distance(cluster[i],cluster[j])
            e = lj(r)
            energy += e
    return energy