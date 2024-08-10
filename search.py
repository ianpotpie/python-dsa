import argparse
import random
import time

import matplotlib.pyplot as plt
import numpy
from scipy import stats

# Search algorithms


def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1


def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] < target:
            low = mid + 1
        elif arr[mid] > target:
            high = mid - 1
        else:
            return mid
    return -1


def binary_search_recursive(arr, target):
    if len(arr) == 0:
        return -1

    mid = len(arr) // 2
    if arr[mid] < target:
        return (mid + 1) + binary_search_recursive(arr[mid + 1 :], target)
    elif arr[mid] > target:
        return binary_search_recursive(arr[:mid], target)
    else:
        return mid


def ternary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        partition_size = (high - low) // 3
        mid1 = low + partition_size
        mid2 = high - partition_size
        if arr[mid1] == target:
            return mid1
        if arr[mid2] == target:
            return mid2
        if arr[mid1] > target:
            high = mid1 - 1
        elif arr[mid2] < target:
            low = mid2 + 1
        else:
            low = mid1 + 1
            high = mid2 - 1
    return -1


def ternary_search_recursive(arr, target):
    if len(arr) == 0:
        return -1

    partition_size = len(arr) // 3
    mid1 = partition_size
    mid2 = 2 * partition_size
    if arr[mid1] == target:
        return mid1
    if arr[mid2] == target:
        return mid2
    if arr[mid1] > target:
        return mid1 + ternary_search_recursive(arr[:mid1], target)
    elif arr[mid2] < target:
        return (mid2 + 1) + ternary_search_recursive(arr[mid2 + 1 :], target)
    else:
        return (mid1 + 1) + ternary_search_recursive(arr[mid1 + 1 : mid2], target)


def exponential_search(arr, target):
    if len(arr) == 0:
        return -1

    if arr[0] == target:
        return 0

    i = 1
    while i < len(arr) and arr[i] <= target:
        i *= 2

    return binary_search_recursive(arr[: min(i + 1, len(arr))], target)


def jump_search(arr, target):
    if len(arr) == 0:
        return -1

    n = len(arr)
    jmp = int(n**0.5)
    step = jmp
    prev = 0
    while arr[min(step, n) - 1] < target:
        prev = step
        step += jmp
        if prev >= n:
            return -1

    while arr[prev] < target:
        prev += 1
        if prev == min(step, n):
            return -1

    if arr[prev] == target:
        return prev

    return -1


def interpolation_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low < high and arr[low] <= target < arr[high]:
        mid = low + ((target - arr[low]) * (high - low)) // (arr[high] - arr[low])
        if arr[mid] < target:
            low = mid + 1
        elif arr[mid] > target:
            high = mid - 1
        else:
            return mid
    return high


def fibonacci_search(arr, target):
    n = len(arr)
    left, right = 0, 1
    while (left + right) < n:
        left, right = right, right + left  # increment by 1 in the fibonacci sequence

    offset = -1

    while (left + right) > 1:
        i = min(offset + left, n - 1)
        left, right = right - left, left
        if arr[i] < target:
            offset = i
        elif arr[i] > target:
            left, right = right - left, left
        else:
            return i

    return offset + 1 if offset + 1 < n and arr[offset + 1] == target else offset + 1


# Timing functions


fns_by_name = {
    "linear_search": linear_search,
    "binary_search": binary_search,
    "binary_search_recursive": binary_search_recursive,
    "ternary_search": ternary_search,
    "ternary_search_recursive": ternary_search_recursive,
    "exponential_search": exponential_search,
    "jump_search": jump_search,
    "interpolation_search": interpolation_search,
    "fibonacci_search": fibonacci_search,
}


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        "-t",
        type=str,
        nargs="*",
        default=[fn for fn in fns_by_name],
        help="Algorithms to include in testing",
    )

    parser.add_argument(
        "--min",
        "-s",
        type=int,
        default=0,
        help="Start list size",
    )

    parser.add_argument(
        "--max",
        "-e",
        type=int,
        default=400,
        help="End list size",
    )

    parser.add_argument(
        "--resolution",
        "-r",
        type=int,
        default=50,
        help="Number of list sizes to test",
    )

    parser.add_argument(
        "--n_trials",
        "-n",
        type=int,
        default=50,
        help="Number of trials to average over",
    )

    parser.add_argument(
        "--metric",
        "-m",
        type=str,
        default="median",
        help="Metric to use for timing",
    )

    parser.add_argument(
        "--distribution",
        "-d",
        type=str,
        default="randint",
        help="Distribution of random numbers",
    )

    parser.add_argument(
        "--params",
        "-p",
        type=float,
        nargs="*",
        default=[0, 1000],
        help="Parameters for the random number distribution",
    )

    parser.add_argument(
        "--blacklist",
        "-b",
        type=str,
        nargs="*",
        default=["bogo_sort"],
        help="Algorithms to exclude from testing",
    )

    return parser.parse_args()


def time_functions(fns_by_name, n_trials, list_sizes, rand_fn, metric):
    fn_times = {name: len(list_sizes) * [[]] for name in fns_by_name}

    for i, list_size in enumerate(list_sizes):
        for j in range(n_trials):
            j_justified = str(j + 1).rjust(len(str(n_trials)), " ")
            i_justified = str(i + 1).rjust(len(str(len(list_sizes))), " ")
            size_justified = str(list_size).rjust(len(str(list_sizes[-1])), " ")
            print(
                f"\33[2K\rTiming trial {j_justified}/{n_trials} on list size {i_justified}/{len(list_sizes)} (curr: {size_justified}, max: {list_sizes[-1]})",
                end="",
            )

            lst = [rand_fn() for _ in range(list_size)]
            tgt = 0 if list_size == 0 else random.choice(lst)
            lst.sort()
            for name, fn in fns_by_name.items():
                cpy = lst.copy()
                start = time.perf_counter()
                fn(cpy, tgt)
                end = time.perf_counter()
                fn_times[name][i].append(end - start)

        for fn in fn_times:
            fn_times[fn][i] = metric(fn_times[fn][i])

    return fn_times


def plot_times(list_sizes, fn_times, metric):
    for name, times in fn_times.items():
        plt.plot(list_sizes, times, label=name)
    plt.ylabel(f"{metric} time (s)")
    plt.xlabel("list size")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    args = get_args()
    args.params = [int(param) if param.is_integer() else param for param in args.params]
    print("Running with args:")
    for arg, val in vars(args).items():
        if isinstance(val, list):
            prefix = len(f"    {arg}: ")
            separator = ", \n" + " " * prefix
            val = separator.join(str(element) for element in val)
        print(f"    {arg}: {val}")
    fns = {
        name: fn
        for name, fn in fns_by_name.items()
        if name in args.test and name not in args.blacklist
    }
    step = (args.max - args.min) // args.resolution
    list_sizes = [
        round(i * (args.max - args.min) / args.resolution + args.min)
        for i in range(args.resolution + 1)
    ]
    distribution = getattr(stats, args.distribution)(*args.params)
    rand_fn = lambda: distribution.rvs()
    metric = getattr(numpy, args.metric)

    print("\nTiming functions")
    fn_times = time_functions(fns, args.n_trials, list_sizes, rand_fn, metric)
    plot_times(list_sizes, fn_times, args.metric)
    print()
