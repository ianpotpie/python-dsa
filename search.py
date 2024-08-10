import random
import time

import matplotlib.pyplot as plt

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


# Timing functions


def time_functions(fns_by_name, n_trials, list_sizes, rand_fn):
    fn_times = {name: [] for name in fns_by_name}
    for name, fn in fns_by_name.items():
        avg_times = []
        for i, list_size in enumerate(list_sizes):
            print(f"\33[2K\rTiming {name} with list {i+1}/{len(list_sizes)}", end="")
            times = []
            for _ in range(n_trials):
                lst = [rand_fn() for _ in range(list_size)]
                tgt = lst[0]
                lst = sorted(lst)
                start = time.perf_counter()
                fn(lst, tgt)
                end = time.perf_counter()
                times.append(end - start)
            avg_times.append(sum(times) / n_trials)
        print()
        fn_times[name] = avg_times
    return fn_times


def plot_times(list_sizes, fn_times):
    for name, times in fn_times.items():
        plt.plot(list_sizes, times, label=name)
    plt.ylabel("Average Time (s)")
    plt.xlabel("List Size")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    list_sizes = [sz for sz in range(1, 200, 1)]
    n_trials = 100
    rand_fn = lambda: random.randint(0, 1000)
    fns_by_name = {
        "linear_search": linear_search,
        "binary_search": binary_search,
    }

    fn_times = time_functions(fns_by_name, n_trials, list_sizes, rand_fn)
    plot_times(list_sizes, fn_times)
