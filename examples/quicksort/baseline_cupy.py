import cupy as cp

def quicksort(array, out=None):
    if len(array) < 2:
        if out:
            out[:] = array
        return

    pivot = int(array[len(array) - 1])  # without type conversion it's a view, not copy
    left_mask = (array < pivot)
    left = array[left_mask]
    right_mask = ~left_mask
    right_mask[len(array) - 1] = False
    right = array[right_mask]
    # assert len(array) == len(left) + 1 + len(right)

    out = out or array
    if len(left):
        quicksort(left)
        xp.assignment(out, cp.arange(len(left)), left, None)
    if len(right):
        quicksort(right)
        xp.assignment(out, cp.arange(len(left) + 1, len(array)), right, None)
    out[len(left)] = pivot


def main(args):
    # np.random.seed(10)
    # cp.random.seed(10)

    cupy_list_in = []
    cupy_list_out = []
    for i in range(args.n):
        with cp.cuda.Device(i):
            random_array = cp.random.randint(0, 100, size=args.m).astype(cp.int32)
            cupy_list_in.append(random_array)
            cupy_list_out.append(cp.zeros_like(random_array))

    x_in = xp.array(cupy_list_in, axis=0)
    x_out = xp.array(cupy_list_out, axis=0)

    from timeit import timeit
    print(timeit(lambda:
        quicksort(x_in, out=x_out)
    , number=3)/3)

    print("origin:", x_in)
    print("sorted:", x_out)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=int, default=3, help="Size of array per GPU.")
    parser.add_argument("-n", type=int, default=2, help="Number of GPUs.")
    args = parser.parse_args()
    main(args)