import sys
import numpy as np
import scipy.optimize as opt
import os
import warnings

warnings.filterwarnings('ignore')


def make_cluster(natoms, radius=20, seed=1981):
    np.random.seed(seed)
    arr = np.random.normal(0, radius, size=(natoms,3))-0.5
    return arr


def potential(r2, epsilon=1., sigma=1.):
    sr6 = (sigma/r2)**3
    return 4.*epsilon*(sr6**2 - sr6)


def gradient(r2, epsilon=1., sigma=1.):
    sr2 = (sigma/r2)
    return -24.*epsilon*(2.*sr2**7 - sr2**4)


def distance_matrix(cluster):
    diff = cluster[:, np.newaxis, :] - cluster[np.newaxis, :, :]
    return diff


def lj(cluster, do_forces=True, *parameters):
    if cluster.ndim == 1:
        cluster = cluster.reshape(-1,3)

    diff = distance_matrix(cluster)
    r2 = (diff**2).sum(-1)

    energy = np.nansum(potential(r2,*parameters))/2.

    if do_forces:
        forces = np.nansum(gradient(r2, *parameters)[:,:,np.newaxis]*diff,axis=0)
        return energy, forces
    else:
        return energy


def write(cluster, filename='optimized.xyz', append=False):
    mode = 'ab' if append else 'wb'
    with open(filename, mode) as f:
        np.savetxt(f, cluster, header='{:d}\n'.format(cluster.shape[0]),
                   comments='', fmt='Ar %12.8f %12.8f %12.8f')

def optimize(cluster, *parameters):
    ret = opt.minimize(lambda x,*args:lj(x,*args)[0], cluster, jac=False,
                       args=parameters, method='BFGS', options={'disp':True})
    return ret.x.reshape(-1,3)

def main(natoms=100):
    print('A {:d} atom cluster'.format(natoms))
    atoms = make_cluster(natoms)
    energy = lj(atoms, do_forces=False)
    print('  Total energy: {:.4f}'.format(energy))

if __name__ == '__main__':
    try:
        main(int(sys.argv[1]))
    except IndexError:
        main()
