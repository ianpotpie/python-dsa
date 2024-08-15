import argparse
import random
import time

import matplotlib.pyplot as plt
import numpy
from scipy import stats

# Simple sorting algorithms fast on small lists but slow on large lists


def bogo_sort(arr):
    while any(arr[i] > arr[i + 1] for i in range(len(arr) - 1)):
        random.shuffle(arr)


def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]


def cycle_sort(arr: list) -> None:
    """
    Perform an in-place cycle sort on the input array.

    Cycle sort is an in-place, unstable sorting algorithm that is particularly useful
    when memory writes are costly. It minimizes the number of memory writes to sort
    an array.

    Args:
        arr (list): The input list to be sorted.

    Returns:
        list: The sorted list (note that the input list is sorted in-place).

    Time complexity: O(n^2) in all cases, where n is the number of elements.
    Space complexity: O(1) as it's an in-place sorting algorithm.

    Example:
    >>> arr = [5, 2, 1, 8, 4, 7, 6, 3]
    >>> cycle_sort(arr)
    [1, 2, 3, 4, 5, 6, 7, 8]
    """
    # loop through the array to find cycles
    for cycle_start in range(len(arr) - 1):
        item = arr[cycle_start]

        # find where to put the item
        pos = cycle_start
        for i in range(cycle_start + 1, len(arr)):
            if arr[i] < item:
                pos += 1

        # the item is already in the correct position
        if pos == cycle_start:
            continue

        # put the item where it goes (after any duplicates)
        while item == arr[pos]:
            pos += 1
        arr[pos], item = item, arr[pos]

        # rotate the rest of the cycle
        while pos != cycle_start:
            pos = cycle_start
            for i in range(cycle_start + 1, len(arr)):
                if arr[i] < item:
                    pos += 1

            while item == arr[pos]:
                pos += 1
            arr[pos], item = item, arr[pos]


def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]


def insertion_sort(arr):
    for i in range(1, len(arr)):
        j = i
        while j > 0 and arr[j] < arr[j - 1]:
            arr[j], arr[j - 1] = arr[j - 1], arr[j]
            j -= 1


def shell_sort(arr):
    n = len(arr)
    gap = n // 2

    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2

    return arr


# Relatively fast and simple recursive sorting algorithms


def merge_sort_simple(arr):
    if len(arr) <= 1:
        return

    def merge(left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1

        result.extend(left[i:])
        result.extend(right[j:])
        return result

    middle = len(arr) // 2
    left, right = arr[:middle], arr[middle:]
    merge_sort_simple(left)
    merge_sort_simple(right)

    for i, val in enumerate(merge(left, right)):
        arr[i] = val


def merge_sort(arr):
    def merge(l, m, r):
        len1, len2 = m - l + 1, r - m
        left, right = arr[l : m + 1], arr[m + 1 : r + 1]

        i, j, k = 0, 0, l
        while i < len1 and j < len2:
            if left[i] <= right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1

        while i < len1:
            arr[k] = left[i]
            i += 1
            k += 1

        while j < len2:
            arr[k] = right[j]
            j += 1
            k += 1

    def sort(start, end):
        if start < end:
            mid = (start + end) // 2
            sort(start, mid)
            sort(mid + 1, end)
            merge(start, mid, end)

    sort(0, len(arr) - 1)


def quick_sort_simple(arr):
    if len(arr) <= 1:
        return

    pivot = arr[-1]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    quick_sort_simple(left)
    quick_sort_simple(right)

    for i, val in enumerate(left + middle + right):
        arr[i] = val


def quick_sort(arr):
    def partition(low, high):
        pivot = arr[high]
        i = low - 1

        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]

        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    def quick_sort_helper(low, high):
        if low < high:
            partition_index = partition(low, high)
            quick_sort_helper(low, partition_index - 1)
            quick_sort_helper(partition_index + 1, high)

    quick_sort_helper(0, len(arr) - 1)


# Fast and complex sorting algorithms


def heap_sort(arr):
    def heapify(arr, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[i] < arr[left]:
            largest = left

        if right < n and arr[largest] < arr[right]:
            largest = right

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)

    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)


def tim_sort(arr, min_run=32):
    def insertion_sort(left, right):
        for i in range(left + 1, right + 1):
            key = arr[i]
            j = i - 1
            while j >= left and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

    def merge(l, m, r):
        len1, len2 = m - l + 1, r - m
        left, right = arr[l : m + 1], arr[m + 1 : r + 1]

        i, j, k = 0, 0, l
        while i < len1 and j < len2:
            if left[i] <= right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1

        while i < len1:
            arr[k] = left[i]
            i += 1
            k += 1

        while j < len2:
            arr[k] = right[j]
            j += 1
            k += 1

    n = len(arr)
    for i in range(0, n, min_run):
        insertion_sort(i, min((i + min_run - 1), n - 1))

    size = min_run
    while size < n:
        for left in range(0, n, 2 * size):
            mid = min(n - 1, left + size - 1)
            right = min((left + 2 * size - 1), (n - 1))
            merge(left, mid, right)
        size *= 2


def tree_sort(arr):
    if len(arr) <= 1:
        return

    class Node:
        def __init__(self, val):
            self.val = val
            self.left = None
            self.right = None

        def insert(self, val):
            if val < self.val:
                if self.left is None:
                    self.left = Node(val)
                else:
                    self.left.insert(val)
            else:
                if self.right is None:
                    self.right = Node(val)
                else:
                    self.right.insert(val)

        def inorder(self):
            if self.left:
                yield from self.left.inorder()
            yield self.val
            if self.right:
                yield from self.right.inorder()

    root = Node(arr[0])
    for val in arr[1:]:
        root.insert(val)

    for i, val in enumerate(root.inorder()):
        arr[i] = val


# Specialized sorting algorithms


def counting_sort(arr):
    if len(arr) <= 1:
        return

    max_val = max(arr)
    min_val = min(arr)
    counts = [0] * (max_val - min_val + 1)

    for val in arr:
        counts[val - min_val] += 1

    for i in range(1, len(counts)):
        counts[i] += counts[i - 1]

    output = [0] * len(arr)
    for val in reversed(arr):
        output[counts[val - min_val] - 1] = val
        counts[val - min_val] -= 1

    for i, val in enumerate(output):
        arr[i] = val


def radix_sort(arr, radix=10):
    if len(arr) <= 1:
        return

    def counting_sort(arr, exp):
        n = len(arr)
        output = [0] * n
        count = [0] * radix

        for i in range(n):
            index = arr[i] // exp
            count[index % radix] += 1

        for i in range(1, len(count)):
            count[i] += count[i - 1]

        i = n - 1
        while i >= 0:
            index = arr[i] // exp
            output[count[index % radix] - 1] = arr[i]
            count[index % radix] -= 1
            i -= 1

        for i in range(n):
            arr[i] = output[i]

    max_val = max(arr, key=abs)
    exp = 1
    negatives = [-x for x in arr if x < 0]
    positives = [x for x in arr if x >= 0]
    while max_val // exp > 0:
        counting_sort(negatives, exp)
        counting_sort(positives, exp)
        exp *= radix

    negatives = [-x for x in reversed(negatives)]
    for i, val in enumerate(negatives + positives):
        arr[i] = val


def bucket_sort(arr, n_buckets=10, sort_fn=None):
    if sort_fn is None:
        sort_fn = lambda x: bucket_sort(x, n_buckets, sort_fn=sort_fn)

    if len(arr) <= 1:
        return

    min_val, max_val = min(arr), max(arr)
    bucket_size = (max_val - min_val) / n_buckets
    output = []

    if max_val == min_val:
        return

    buckets = [[] for _ in range(n_buckets)]
    for val in arr:
        index = int((val - min_val) // bucket_size)
        index = index if index < n_buckets else n_buckets - 1
        buckets[index].append(val)

    for bucket in buckets:
        sort_fn(bucket)
        output.extend(bucket)

    for i, val in enumerate(output):
        arr[i] = val


# Timing functions

fns_by_name = {
    "bogo_sort": bogo_sort,
    "bubble_sort": bubble_sort,
    "insertion_sort": insertion_sort,
    "selection_sort": selection_sort,
    "merge_sort_simple": merge_sort_simple,
    "heap_sort": heap_sort,
    "tree_sort": tree_sort,
    "shell_sort": shell_sort,
    "quick_sort": quick_sort,
    "merge_sort": merge_sort,
    "tim_sort": tim_sort,
    "quick_sort_simple": quick_sort_simple,
    "radix_sort": radix_sort,
    "bucket_sort": bucket_sort,
    "counting_sort": counting_sort,
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
            for name, fn in fns_by_name.items():
                cpy = lst.copy()
                start = time.perf_counter()
                fn(cpy)
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
