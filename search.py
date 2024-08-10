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
    if arr[0] == target:
        return 0

    i = 1
    while i < len(arr) and arr[i] <= target:
        i *= 2

    return binary_search_recursive(arr[: min(i + 1, len(arr))], target)


def jump_search(arr, target):
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
    while low <= high and arr[low] <= target <= arr[high]:
        mid = low + ((target - arr[low]) * (high - low)) // (arr[high] - arr[low])
        if arr[mid] < target:
            low = mid + 1
        elif arr[mid] > target:
            high = mid - 1
        else:
            return mid
    return -1


def fibonacci_search(arr, target):
    n = len(arr)
    left, right = 0, 1
    while (left + right) < n:
        right, left = left, right + left  # increment by 1 in the fibonacci sequence

    offset = -1

    while (left + right) > 1:
        print(offset, left, right)
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
    lst = [i for i in range(50)]
    for i in range(50):
        print(fibonacci_search(lst, i))
    exit()

    list_sizes = [sz for sz in range(1, 200, 1)]
    n_trials = 100
    rand_fn = lambda: random.randint(0, 1000)
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

    fn_times = time_functions(fns_by_name, n_trials, list_sizes, rand_fn)
    plot_times(list_sizes, fn_times)
