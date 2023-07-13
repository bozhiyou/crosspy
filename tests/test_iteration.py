import crosspy as xp
from crosspy import cpu, gpu
import numpy
import cupy
import asyncio
from inspect import isgenerator


def test_block_view():
    x = xp.empty([1,1,1], device=[gpu(1), cpu(0), gpu(0)])

    y = x.block_view()
    assert isgenerator(y)
    list(y)
    
    y = x.block_view(lambda x: x)
    assert isgenerator(y)
    list(y)

    async def alambda(x):
        return x
    y = x.block_view(alambda)
    assert isgenerator(y)
    list(y)

    async def wrap():
        y = x.block_view(lambda x: x)
        assert isgenerator(y)
        list(y)
    asyncio.run(wrap())

    async def awrap():
        y = x.block_view(alambda)
        assert isgenerator(y)
        for c in y:
            await c
    asyncio.run(awrap())


async def test_async_sort(n):
    a = []
    for i in range(n):
        with cupy.cuda.Device(i):
            a.append(cupy.random.rand(2**28//n))

    async def asyncsort(p):
        cupy.sort(p)
        await asyncio.sleep(0)

    await asyncio.gather(*(asyncsort(x) for x in a))

if __name__ == '__main__':
    test_block_view()

    # import time
    # t = time.perf_counter()
    # asyncio.run(test_async_sort(2))
    # print(time.perf_counter() - t)