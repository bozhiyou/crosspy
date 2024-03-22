import asyncio
import cupy as cp
import time
import types

N = 2**28
Ng = 2

arrays = []
for i in range(Ng):
    with cp.cuda.Device(i):
        arrays.append(cp.arange(N // Ng, -1, -1))

time.sleep(1)

async def sort_single(x):
    with x.device:
        with cp.cuda.Stream(non_blocking=True):
            return cp.sort(x)

async def main0():
    T = [asyncio.create_task(sort_single(x)) for x in arrays]
    done, pending = await asyncio.wait(T)
    return T

async def main1():
    T = await asyncio.gather(*(sort_single(x) for x in arrays))
    return T

t = time.perf_counter()
res = asyncio.run(main0())
print(time.perf_counter() - t)
print(res)