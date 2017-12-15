from dask.distributed import Executor
import numpy as np
import dask.array as da
import dask.bag as db
import dask
import os

NCPUS = os.cpu_count()


def do_compute(seed, size=int(4e4), radius=300):
    with dask.set_options(get=dask.threaded.get):
        #da.random.seed(seed)
        #arr = (da.random.normal(0.01, 1, (size,3), chunks=size//24)-0.5)*radius
        np.random.seed(seed)
        c = (np.random.normal(0.01, 1, (size,3))-0.5)*radius
        arr = da.from_array(c, chunks=c.shape[0]//NCPUS)

        diff = arr[:, np.newaxis, :] - arr[np.newaxis, :, :]
        mat = da.sqrt((diff*diff).sum(-1))

        inv6 = (1./mat)**6
        pot = 4.*(inv6*inv6 - inv6)
        e = da.nansum(pot)/2.

        return e.compute(num_workers=NCPUS)


e = Executor('127.0.0.1:8786')
e.restart()

futures = e.map(do_compute, range(2))
out = e.gather(futures)
print(out)
