import torch as cp
from time import perf_counter as time
import argparse 
import nvtx

parser = argparse.ArgumentParser()
parser.add_argument('-ngpus', type=int, default=1)
parser.add_argument('-n', type=int, default=10000000)
parser.add_argument('-repeat', type=int, default=3)
args = parser.parse_args()

from multiprocessing.pool import ThreadPool as WorkerPool
#cp.cuda.set_allocator(cp.cuda.MemoryAsyncPool().malloc)

#Initialize Data
X = []
A = []
for i in range(args.ngpus):
    with cp.cuda.device(i) as device:
        nvtx.push_range(message=f"Create Data on Device {i}", color="blue", domain="app")
        stream = cp.cuda.Stream()
        with cp.cuda.stream(stream):
            X.append(cp.arange(args.n, device=cp.device(f"cuda:{i}")))
            A.append([])
            stream.synchronize()
        nvtx.pop_range(domain="app")


def t1(i):
    nvtx.push_range(message=f"Start Thread {i}", color="yellow", domain="app")

    for k in range(args.repeat):
        #sleep_nogil(1)
        with cp.cuda.device(i):
            stream = cp.cuda.Stream()
            with cp.cuda.stream(stream):

                nvtx.push_range(message=f"Sort {k} on Device {i}", color="red", domain="app")
                t1 =time()
                A[i], _ = cp.sort(X[i])
                #A[i] = X[i] @ X[i].T
                stream.synchronize()
                t2=time()
                nvtx.pop_range(domain="app")
                print("Device ", X[i].device, A[i].device, "len", len(X[i]), "time=%.4E"%(t2-t1), flush=True)
    nvtx.pop_range(domain="app")


pool = WorkerPool(args.ngpus)
start_t = time()
results = pool.map(t1, [i for i in range(args.ngpus)])
pool.close()
pool.join()
end_t = time()
print("Total: ", end_t - start_t, flush=True)