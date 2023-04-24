import crosspy as xp
from crosspy import gpu
import numpy as np
from time import time
import cupy as cp

ngpus = cp.cuda.runtime.getDeviceCount()
repetitions = int(sys.argv[1])

# set up two n x n arrays to multiply together.
# n is chosen so that all three can be
# stored within the memory of a single GPU
# so that strong scaling numbers make sense.
n = 32000

blocks = ngpus
block_size = n // ngpus
h_ordr = 'C'
d_ordr = 'F'
print("BlockSize: ", block_size, "GPUS: ", ngpus)

# Overdecomposing doesn't actually seem to help in this case
# with the current parla runtime. This may be related to
# some weirdness within the scheduler though, so
# we can leave the code for blocks in-place for further
# testing later.

np.random.seed(0)
a_cpu = np.random.rand(n, n).astype(dtype=np.float32, order=h_ordr)
b_cpu = np.random.rand(n, n).astype(dtype=np.float32, order=h_ordr)

print("Finished Data Allocation", flush=True)
# Partition the two arrays and set up the
# partitioned array where the result will be stored.
# This could also be done using a parla mapper object.

a_part = []
b_part = []
c_part = []

distribute=True
fixed_placement=False
verbose=False
sync = False

if fixed_placement:
    loc = gpu(i%ngpus)
else:
    loc = gpu

time_list = list()

# Start all operans from CPU memory.
for i in range(blocks):
    if distribute:
        with cp.cuda.Device(i):
            a_part.append(cp.asarray(a_cpu[i * block_size : (i + 1) * block_size], order=d_ordr))
            b_part.append(cp.asarray(b_cpu[i * block_size : (i + 1) * block_size], order=d_ordr))
            cp.cuda.stream.get_current_stream().synchronize()

    else:
        a_part.append(a_cpu[i * block_size : (i + 1) * block_size])
        b_part.append(b_cpu[i * block_size : (i + 1) * block_size])


for i in range(blocks):
    c_part.append(list())
    for j in range(blocks):
        c_part[i].append(np.empty((0, 0), dtype=np.float32, order=h_ordr))

#print(len(c_part), len(c_part[0]), c_part[0][0].shape)

for repetition in range(repetitions):

    #reset cblocks to None
    for i in range(blocks):
        for j in range(blocks):
            c_part[i][j] = np.empty((0, 0), dtype=np.float32, order=h_ordr)

    matmul = TaskSpace("matmul")
    cp.sin(cp.asarray([2]))
    cp.cuda.stream.get_current_stream().synchronize()
    start = time.perf_counter()
    for i in range(blocks):
        for j in range(blocks):
            a_block = a_part[i]
            b_block = b_part[j]
            c_block = c_part[i][j]

            memsize = (block_size**2 + block_size*n*2)*4

            # @spawn(matmul[i, j], placement = loc, memory=memsize)
            # def matmul_task():
            current_device = cp.cuda.runtime.getDevice()
            #assert(a_block.device.id == current_device) #check row_cyclic

            a = clone_here(a_block)
            b = clone_here(b_block)
            c = clone_here(c_block)

            stream = cp.cuda.get_current_stream()
            if sync:
                stream.synchronize()

            assert(a.device.id == b.device.id)

            if verbose:
                print(f"+({i}, {j}): ", a.shape, b.shape, c.shape, " | On Device: ", current_device, a.device.id, flush=True)

            local_start = time.perf_counter()
            c = a @ b.T

            if sync:
                stream.synchronize()
            local_end = time.perf_counter()

            c_part[i][j] = c

            if verbose:
                print(f"-({i}, {j}): ", a.shape, b.shape, c.shape, " | Elapsed: ", local_end-local_start, flush=True)

    # await matmul
    stop = time.perf_counter()
    print(f"Iteration {repetition} | Time elapsed: ", stop - start, flush=True)
    time_list.append(stop-start)

mean = np.mean(np.array(time_list))
median = np.median(np.array(time_list))

print(f"Execution Time:: Average = {mean} | Median = {median}", flush=True)

print('kmeans iteration')
tic = time()
for j in range(kmeans_maxit):
    for k in range(nc):
        W = X-C[k,:]
        D[:,k] = cp.sum(W*W,axis=1)

    L = cp.argmin(D,axis=1)
    if j==0:L0=L

    for k in range(nc):
        m = cp.sum(L==k)
        with m.item().device:
            assert m.item() != 0, "cannot handle corner case when centers are repetitive: %s in %s" % (cpos[k], cpos)
        ck = 1/m *cp.sum(X[L==k,:],axis=0)
        C[k,:]=ck

toc = time()       
L0h= xp.asnumpy(L0)
Lh = xp.asnumpy(L)
sc0 =homogeneity_score(L0h, labels_ex)
sc =homogeneity_score(Lh, labels_ex)
print(f'Initial homogeneity score = {sc0:.2}')
print(f'homogeneity score = {sc:.2} (1 is best)')
print(f'Kmeans took {toc-tic:.3} seconds')

    