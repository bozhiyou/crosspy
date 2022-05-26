def operation(a, b):
    return a @ b.T

def schedule():
    @spawn(matmul[i, j], placement = loc, memory=memsize, input=[a_block, b_block], output=[c_block])
    def matmul_task():
        a = a_block.array
        b = b_block.array
        c = c_block.array

        stream = cp.cuda.get_current_stream()
        stream.synchronize()

        assert(a.device.id == b.device.id)
        if verbose:
            print(f"+({i}, {j}): ", a.shape, b.shape, c.shape, " | On Device: ", cp.cuda.runtime.getDevice(), a.device.id, flush=True)
        local_start = time.perf_counter()

        c = operation(a, b)

        if sync:
            stream.synchronize()

        local_end = time.perf_counter()

        c_block.update(c)

        c = c_block.array
        if verbose:
            print(f"-({i}, {j}): ", a.shape, b.shape, c.shape, " | Elapsed: ", local_end-local_start, flush=True)