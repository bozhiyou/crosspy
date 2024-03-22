from parla import Parla, spawn, TaskSpace, sleep_nogil
from parla.cython.device_manager import cpu, gpu
from parla.common.globals import get_current_context

import numpy as np
import cupy as cp
import crosspy as xp
import time
import argparse
import nvtx

from parla.cython.device import PyGPUDevice
import crosspy.device
@crosspy.device.of(PyGPUDevice)
def converter(parlagpu):
    return xp.gpu(parlagpu.id)

ngpus = 2
num_tests = 1
class A: n = 1
args = A()
args.n = 10

def main():


    #divisions = len(get_all_devices())*2
    devs = gpu.devices
    
    divisions = ngpus
    # print(divisions)
    
    # Set up an "n" x "n" grid of values and run
    # "steps" iterations of the 4 point stencil on it.
    n = args.n
    matrix_size = n *n
    steps = 1
    blocks = ngpus
    block_size = n // ngpus
    part_size = n // 2
    # part_width = 2 * n // ngpus
    # part_length = 

    @spawn(placement=cpu)
    async def main_jacobi():
        # Set up two arrays containing the input data.
        # This demo uses the standard technique of computing
        # from one array into another then swapping then
        
        # input and output arrays for the next iteration.
        # These are the two arrays that will be swapped back
        # and forth as input and output.
        
        a0 = np.random.rand(matrix_size).astype('f')  
        a1 = a0.copy()

        # An object that distributes arrays across all the given devices.
        num_part = blocks*part_size
        partition = []
        itr = 0
        for i in range(0, matrix_size, part_size):
            loc = gpu(0)
            if(itr < num_part/2):
                loc = xp.gpu(itr % 2)
            else:
                loc = xp.gpu((itr % 2))
            partition.append(loc)
            itr += 1
        print(partition)
        a0_crosspy = xp.array(a0, distribution=partition, axis=0)
        a1_crosspy = xp.array(a1, distribution=partition, axis=0)
        print(a0_crosspy)
        
        device_to_array_map = {}
        itr = divisions*part_size
        for slice, device_id in a0_crosspy.device_map.items():
            if(device_id.id in device_to_array_map):
                temp = device_to_array_map.get(device_id.id)
            else:
                temp = []
            slice_new = slice[0]
            for i in range(slice_new[0], slice_new[1]):
                temp.append(i)
            # temp.append(slice_new[1] - 1)
            device_to_array_map[device_id.id] = temp
        # print(device_to_array_map)
        # print(a0_crosspy)
        # print(a0_crosspy.devices)

        for k in range(num_tests):

            # Specify which set of blocks is used as input or output
            # (they will be swapped for each iteration).

            in_blocks = a0_crosspy
            out_blocks = a1_crosspy
            # print(in_blocks.devices)
            # print(out_blocks.devices)
            # print(len(in_blocks.devices))
            # print(len(out_blocks.devices))


            # Create a set of labels for the tasks that perform the first
            # Jacobi iteration step.
            tslist = list()
            print("added tlist")
            previous_block_tasks = []
            print("added completed")

            start = time.perf_counter()

            # Now create the tasks for subsequent iteration steps.
            for i in range(steps):

                # Swap input and output blocks for the next step.
                # print("In_blocks len: {}".format(len(in_blocks)))
                # print("Out blocks len: {}".format(len(out_blocks)))
                in_blocks, out_blocks = out_blocks, in_blocks
                # Create a new set of labels for the tasks that do this iteration step.

                current_block_tasks = TaskSpace("block_tasks[{}]".format(i))
                print("added current")
                # Create the tasks to do the i'th iteration.
                # As before, each task needs the following info:
                #  a block index "j"
                #  a "device" where it should execute (supplied by mapper used for partitioning)
                #  the "in_block" of data used as input
                #  the "out_block" to write the output to
                

                for j in range(divisions):

                    # if args.fixed:
                    #     device = mapper.device(j)
                    # else:
                    #     device = devs
                    device = gpu(j)
                    right_boundary = []
                    left_boundary = []
                    up_boundary = []
                    down_boundary = []
                    for i in range(part_size):
                        if(j % 2 == 0):
                            right_boundary.append((j // 2) * part_size * n + part_size + i * n)
                        if(j % 2 != 0):
                            left_boundary.append((j // 2) * part_size * n + (part_size - 1) + i * n)
                        if(j // 2 != 0):
                            up_boundary.append((j % 2) * part_size + (part_size - 1) * n + i)
                        if(j // 2 == 0):
                            down_boundary.append(j * part_size + n * part_size + i)
                    st = time.perf_counter()
                    
                    # Make each task operating on each block depend on the tasks for
                    # that block and its immediate neighbors from the previous iteration.
                    # m_rng = nvtx.start_range(message="task", color="red")
                    # print(previous_block_tasks[max(0, j-1):min(divisions, j+2)])
                    print("added boundaries")
                    @spawn(current_block_tasks[j],
                           dependencies=[previous_block_tasks[max(0, j-1):min(divisions, j+2)]],
                           placement=device)
                    def device_local_jacobi_task():
                        print("+Jacobi Task", i, j, flush=True)
                        # Read boundary values from adjacent blocks in the partition.
                        # This may communicate across device boundaries.

                        stream = cp.cuda.get_current_stream()
                        # data_start = time.perf_counter()
                        
                        # data_end = time.perf_counter()
                        #stream.synchronize()
                        #print("Data: ", data_end - data_start, flush=True)

                        # Run the computation, dispatching to device specific code.


                        #print(in_block.shape, out_block.shape, flush=True)
                        interior = device_to_array_map[j]
                        print(f'interior len: {len(interior)}')

                        s1 = time.perf_counter()
                        # rng = nvtx.start_range(message="copy", color="blue")
                        in_block = in_blocks[interior + right_boundary + down_boundary + left_boundary + up_boundary].to(device)
                        print("In block len: {} {}".format(j, len(in_block)))
                        out_block = out_blocks[interior].to(device)
                        stream.synchronize()
                        e1 = time.perf_counter()
                        t1 = e1 - s1
                        # nvtx.end_range(rng)
                        print("Done copying: {}".format(t1), flush=True)
                        
                        compute_start = time.perf_counter()
                        rng2 = nvtx.start_range(message="jacobi", color="green")
                        # jacobi(j, part_size, in_block, out_block)
                        # out_block = out_block1
                        stream.synchronize()
                        # nvtx.end_range(rng2)
                        compute_end = time.perf_counter()
                        # print("Compute: ", compute_end - compute_start, flush=True)

                # For the next iteration, use the newly created tasks as
                # the tasks from the previous step.

                tslist.append(previous_block_tasks)
                previous_block_tasks = current_block_tasks

            await current_block_tasks
            end = time.perf_counter()
            # nvtx.end_range(m_rng)
            print("Time: ", end - start)

            # This depends on all the tasks from the last iteration step.
            #for j in range(divisions):
            #    start_index = 1 if j > 0 else 0
            #    end_index = -1 if j < divisions - 1 else None  # None indicates the last element of the dimension
            #    copy(a1[mapper.slice(j, len(a1))], out_blocks[j][start_index:end_index])

        del a0
        del a1

if __name__ == '__main__':
    with Parla():
        main()