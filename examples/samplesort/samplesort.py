import numpy as np
import cupy as cp
from parla import Parla
# from parla.tasks import spawn, TaskSpace
# from parla.devices import cpu, gpu
import time
import argparse
import crosspy as xp
import asyncio

async def partitioned_crosspy(global_size: int):
    num_gpu  = cp.cuda.runtime.getDeviceCount()

    async def alloc_cuda(l_sz:int, device_id:int):
        with cp.cuda.Device(device_id):
            return cp.random.uniform(0,1, size=l_sz)
        
    T = [asyncio.create_task(alloc_cuda((((i+1) * global_size) // num_gpu -  (i * global_size) // num_gpu ), i)) for i in range(num_gpu)]
    await asyncio.wait(T)
    return xp.array([t.result() for t in T], axis=0)

async def async_range(count):
    for i in range(count):
        yield(i)
        await asyncio.sleep(0)

async def crosspy_sample_sort(x:xp.array):
    num_gpu   = x.ndevices
    
    # print("before")
    # print(x.device_map)
    # u         = xp.array(list(x.block_view(cp.sort)), axis=0)
    # print("after")
    # print(u.device_map)
    # print(x)
    # print(x.device_map)

    async def local_sort(x, device_id):
        with cp.cuda.Device(device_id):
            return cp.sort(x.blockview[device_id])

    T = [asyncio.create_task(local_sort(x, i)) for i in range(num_gpu)]
    await asyncio.wait(T)
    u = xp.array([t.result() for t in T], axis=0)

    # print(u)
    # print(u.device_map)
    
    sp        = np.zeros(num_gpu * (num_gpu-1))
    u_blkview = u.blockview
    
    async for i in async_range(num_gpu):
        y   = u_blkview[i]
        idx = np.array([(((j+1) * len(y)) // num_gpu)-1 for j in range(num_gpu-1)], dtype=np.int32)
        assert (idx>0).all()
        with cp.cuda.Device(i):
            sp[i * (num_gpu-1) : (i+1) * (num_gpu-1)] = cp.asnumpy(y[idx])

    sp  = np.sort(sp)

    num_splitters = num_gpu-1

    idx = np.array([((i) * len(sp) // num_splitters) + (((i+1) * len(sp) // num_splitters) - ((i) * len(sp) // num_splitters))//2 for i in range(num_splitters)],dtype=np.int32)
    sp  = sp[idx]

    sp_list = []
    async for i in async_range(num_gpu):
        with cp.cuda.Device(i):
            sp_list.append(cp.asarray(sp))
    
    sp = xp.array(sp_list, axis=0)
    sp_blkview = sp.blockview

    # print(sp.device_map)
    # print(u.device_map)
    
    # to compute global ids
    local_counts = np.array([len(u_blkview[i]) for i in range(num_gpu)])
    local_offset = np.append(np.array([0]), local_counts)
    local_offset = np.cumsum(local_offset)[:-1]

    sendcnts  = np.zeros((num_gpu,num_gpu), dtype=np.int32)
    gid_send  = np.zeros(u.shape[0], dtype=np.int32)

    for i in range(num_gpu):
        with cp.cuda.Device(i):
            idx           = cp.asnumpy(cp.where(u_blkview[i]<sp_blkview[i][0])[0]) 
            sendcnts[i,0] = len(idx)
            
            for sp_idx in range(1, num_splitters):
                cond = cp.logical_and(u_blkview[i]>=sp_blkview[i][sp_idx-1],  u_blkview[i]<sp_blkview[i][sp_idx])
                idx  = cp.asnumpy(cp.where(cond)[0])
                sendcnts[i, sp_idx] = len(idx)
        
            idx = cp.asnumpy(cp.where(u_blkview[i]>=sp_blkview[i][num_splitters-1])[0])
            sendcnts[i, num_gpu-1] = len(idx)
        
    #recvcnts = np.transpose(sendcnts)
    arr_list = list()
    async for i in async_range(num_gpu):
        with cp.cuda.Device(i):
            arr_list.append(cp.zeros(np.sum(sendcnts[:,i])))
    
    rbuff         = xp.array(arr_list, axis=0)
    rbuff_blkview = rbuff.blockview

    #print(rbuff.device_map)

    rbuff_counts  = np.array([len(rbuff_blkview[i]) for i in range(num_gpu)])
    rbuff_offset  = np.append(np.array([0]), rbuff_counts)
    rbuff_offset  = np.cumsum(rbuff_offset)[:-1]

    for i in range(num_gpu):
        tmp = np.array([],dtype=np.int32)
        for j in range(num_gpu):
            tmp=np.append(tmp, local_offset[j] + np.array(range(sendcnts[j,i]), dtype=np.int32))

        #print(i, tmp, len(tmp), rbuff_counts[i])
        gid_send[rbuff_offset[i] : rbuff_offset[i] + rbuff_counts[i]] = tmp

    gid_recv = np.array(range(u.shape[0]),dtype=np.int64)
    assignment = await xp.alltoall(rbuff, gid_recv, u, gid_send.astype(np.int64, copy=False))
    await assignment()
    print(rbuff)
    
    return rbuff

al = []
for i in range(2):
    with cp.cuda.Device(i):
        al.append(cp.random.rand(5))
asyncio.run(crosspy_sample_sort(xp.array(al, axis=0)))