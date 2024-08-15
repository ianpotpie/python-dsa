import argparse
import multiprocessing as mp
import random
import time
import tracemalloc
from collections.abc import Callable
from functools import partial
from typing import Any, Protocol, TypeVar

import matplotlib.pyplot as plt
import numpy
from scipy import stats


class Comparable(Protocol):
    def __lt__(self, other: Any) -> bool: ...
    def __le__(self, other: Any) -> bool: ...
    def __eq__(self, other: Any) -> bool: ...
    def __ge__(self, other: Any) -> bool: ...
    def __gt__(self, other: Any) -> bool: ...


T = TypeVar("T", bound=Comparable)


def bogo_sort(arr: list[T], inplace=True) -> list[T]:
    """
    Perform bogo sort on the input list.

    Bogo sort is an in-place, unstable sorting algorithm that shuffles the list
    until it is sorted. It is not a practical sorting algorithm and is used primarily
    for educational purposes.

    Args:
        arr (list[T]): The input list to be sorted.
        inplace (bool): Whether to sort the list in-place.

    Returns:
        list[T]: The sorted list.

    Time complexity:
        Best case: O(n) where n is the number of elements.
        Worst case: O(∞) as the algorithm may never terminate.
        Average case: O((n+1)!) where n is the number of elements.

    Space complexity:
        O(1) as sorting is done in-place.
    """
    if not inplace:
        arr = arr.copy()

    # Shuffle the list until it is sorted
    n = len(arr)
    while any(arr[i] > arr[i + 1] for i in range(n - 1)):
        random.shuffle(arr)

    return arr


def cycle_sort(arr: list[T], inplace=True) -> list[T]:
    """
    Perform cycle sort on the input list.

    Cycle sort is an in-place, unstable sorting algorithm that is optimal in
    terms of the number of memory writes. It's based on the idea that the permutation
    to be sorted can be decomposed into cycles, and each cycle can be rotated to find
    its correct position.

    Args:
        arr (list[T]): The input list to be sorted.
        inplace (bool): Whether to sort the list in-place.

    Returns:
        list[T]: The sorted list.

    Time complexity:
        Best case: O(n^2) where n is the number of elements.
        Worst case: O(n^2) where n is the number of elements.
        Average case: O(n^2) where n is the number of elements.

    Space complexity:
        O(1) as sorting is done in-place.
    """
    if not inplace:
        arr = arr.copy()

    # loop through the array to find cycles
    n = len(arr)
    for cycle_start in range(n - 1):
        item = arr[cycle_start]

        # find where to put the item
        pos = cycle_start
        for i in range(cycle_start + 1, n):
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
            for i in range(cycle_start + 1, n):
                if arr[i] < item:
                    pos += 1

            while item == arr[pos]:
                pos += 1
            arr[pos], item = item, arr[pos]

    return arr


def bubble_sort(arr: list[T], inplace=True) -> list[T]:
    """
    Perform bubble sort on the input list.

    Bubble sort is an in-place, stable sorting algorithm that repeatedly steps
    through the list, compares adjacent elements and swaps them if they are in
    the wrong order. The pass through the list is repeated until the list is sorted.

    Args:
        arr (list[T]): The input list to be sorted.
        inplace (bool): Whether to sort the list in-place.

    Returns:
        list[T]: The sorted list.

    Time complexity:
        Best case: O(n) where n is the number of elements (when the list is already sorted).
        Worst case: O(n^2) where n is the number of elements.
        Average case: O(n^2) where n is the number of elements.

    Space complexity:
        O(1) as sorting is done in-place.
    """
    if not inplace:
        arr = arr.copy()

    n = len(arr)
    for i in range(n):
        # Flag to check if any swapping is done in the inner loop
        swapped = False

        for j in range(n - i - 1):
            # Swap if the element is greater than the next element
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True

        # If no swapping occurred, array is already sorted
        if not swapped:
            break

    return arr


def comb_sort(arr: list[T], inplace=True) -> list[T]:
    """
    Perform comb sort on the input list.

    Comb sort is an improvement over bubble sort. It eliminates small values
    near the end of the list by using a gap larger than 1. The gap starts with
    a large value and shrinks by a factor of 1.3 in every iteration until it reaches 1.

    Args:
        arr (list[T]): The input list to be sorted.
        inplace (bool): Whether to sort the list in-place.

    Returns:
        list[T]: The sorted list.

    Time complexity:
        Best case: O(n log n) where n is the number of elements.
        Worst case: O(n^2) where n is the number of elements.
        Average case: O(n^2 / 2^p) where n is the number of elements and p is the number of increments.

    Space complexity:
        O(1) as sorting is done in-place.
    """
    if not inplace:
        arr = arr.copy()

    n = len(arr)
    gap = n
    shrink = 1.3
    sorted = False

    while not sorted:
        # Update the gap
        gap = int(gap / shrink)
        if gap <= 1:
            gap = 1
            sorted = True

        # Compare elements with the current gap
        for i in range(n - gap):
            if arr[i] > arr[i + gap]:
                arr[i], arr[i + gap] = arr[i + gap], arr[i]
                sorted = False

    return arr


def insertion_sort(arr: list[T], inplace=True) -> list[T]:
    """
    Perform insertion sort on the input list.

    Insertion sort is an in-place, stable sorting algorithm that builds the sorted array
    one item at a time. It is much less efficient on large lists than more advanced algorithms
    such as quicksort, heapsort, or merge sort.

    Args:
        arr (list[T]): The input list to be sorted.
        inplace (bool): Whether to sort the list in-place.

    Returns:
        list[T]: The sorted list.

    Time complexity:
        Best case: O(n) where n is the number of elements (when the list is already sorted).
        Worst case: O(n^2) where n is the number of elements.
        Average case: O(n^2) where n is the number of elements.

    Space complexity:
        O(1) as sorting is done in-place.
    """
    if not inplace:
        arr = arr.copy()

    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1

        # Move elements of arr[0..i-1], that are greater than key, to one position ahead
        # of their current position
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = key

    return arr


def gnome_sort(arr: list[T], inplace=True) -> list[T]:
    """
    Perform gnome sort on the input list.

    Gnome sort is an in-place, stable sorting algorithm that is similar to insertion sort,
    but moving an element to its proper place is accomplished by a series of swaps, as in bubble sort.
    It's called "gnome sort" because it's the way a garden gnome sorts a line of flower pots.

    Args:
        arr (list[T]): The input list to be sorted.
        inplace (bool): Whether to sort the list in-place.

    Returns:
        list[T]: The sorted list.

    Time complexity:
        Best case: O(n) where n is the number of elements (when the list is already sorted).
        Worst case: O(n^2) where n is the number of elements.
        Average case: O(n^2) where n is the number of elements.

    Space complexity:
        O(1) as sorting is done in-place.
    """
    if not inplace:
        arr = arr.copy()

    n = len(arr)
    index = 0

    while index < n:
        if index == 0:
            index += 1
        if arr[index] >= arr[index - 1]:
            index += 1
        else:
            arr[index], arr[index - 1] = arr[index - 1], arr[index]
            index -= 1

    return arr


def selection_sort(arr: list[T], inplace=True) -> list[T]:
    """
    Perform selection sort on the input list.

    Selection sort is an in-place, unstable sorting algorithm that divides the input list
    into two parts: a sorted part and an unsorted part. The algorithm repeatedly selects
    the smallest (or largest, depending on the sorting order) element from the unsorted
    part and moves it to the end of the sorted part.

    Args:
        arr (list[T]): The input list to be sorted.
        inplace (bool): Whether to sort the list in-place.

    Returns:
        list[T]: The sorted list.

    Time complexity:
        Best case: O(n^2) where n is the number of elements.
        Worst case: O(n^2) where n is the number of elements.
        Average case: O(n^2) where n is the number of elements.

    Space complexity:
        O(1) as sorting is done in-place.
    """
    if not inplace:
        arr = arr.copy()

    n = len(arr)
    for i in range(n):
        # Assume the minimum is the first element
        min_index = i

        # Find the minimum element in the remaining unsorted part
        for j in range(i + 1, n):
            if arr[j] < arr[min_index]:
                min_index = j

        # Swap the found minimum element with the first element
        arr[i], arr[min_index] = arr[min_index], arr[i]

    return arr


def shell_sort(arr: list[T], inplace=True) -> list[T]:
    """
    Perform shell sort on the input list.

    Shell sort is an in-place, unstable sorting algorithm that is a generalization
    of insertion sort. It allows the exchange of items that are far apart and
    reduces the time complexity compared to bubble sort and insertion sort.

    Args:
        arr (list[T]): The input list to be sorted.
        inplace (bool): Whether to sort the list in-place.

    Returns:
        list[T]: The sorted list.

    Time complexity:
        Best case: O(n log n) where n is the number of elements.
        Worst case: O(n^2) where n is the number of elements.
        Average case: O(n^(3/2)) where n is the number of elements.

    Space complexity:
        O(1) as sorting is done in-place.
    """
    if not inplace:
        arr = arr.copy()

    n = len(arr)
    gap = n // 2

    while gap > 0:
        # Perform a gapped insertion sort for this gap size
        for i in range(gap, n):
            temp = arr[i]
            j = i

            # Shift earlier gap-sorted elements up until the correct location for arr[i] is found
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap

            # Put temp (the original arr[i]) in its correct location
            arr[j] = temp

        gap //= 2

    return arr


def merge_sort(arr: list[T], inplace=True) -> list[T]:
    """
    Perform merge sort on the input list.

    Merge sort is a divide and conquer algorithm that splits the list into smaller
    sublists, sorts them, and then merges them back together.

    Args:
        arr (list[T]): The input list to be sorted.
        inplace (bool): Whether to sort the list in-place.

    Returns:
        list[T]: The sorted list.

    Time complexity:
        Best case: O(n log n) where n is the number of elements.
        Worst case: O(n log n) where n is the number of elements.
        Average case: O(n log n) where n is the number of elements.

    Space complexity:
        O(n) for the temporary arrays used during the merge process.
    """
    if not inplace:
        arr = arr.copy()

    def merge(left: list[T], right: list[T]) -> list[T]:
        merged = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged

    def merge_sort_helper(arr: list[T]) -> list[T]:
        if len(arr) <= 1:
            return arr

        mid = len(arr) // 2
        left = merge_sort_helper(arr[:mid])
        right = merge_sort_helper(arr[mid:])

        return merge(left, right)

    arr[:] = merge_sort_helper(arr)
    return arr


def quicksort(arr: list[T], inplace=True) -> list[T]:
    """
    Perform quicksort on the input list.

    Quicksort is an efficient, in-place, unstable sorting algorithm that uses a
    divide-and-conquer strategy. It works by selecting a 'pivot' element from the
    array and partitioning the other elements into two sub-arrays, according to
    whether they are less than or greater than the pivot.

    Args:
        arr (list[T]): The input list to be sorted.
        inplace (bool): Whether to sort the list in-place.

    Returns:
        list[T]: The sorted list.

    Time complexity:
        Best case: O(n log n) where n is the number of elements.
        Worst case: O(n^2) where n is the number of elements (rare, occurs with poor pivot choices).
        Average case: O(n log n) where n is the number of elements.

    Space complexity:
        O(log n) due to the recursive call stack.
    """
    if not inplace:
        arr = arr.copy()

    def partition(low: int, high: int) -> int:
        pivot = arr[high]
        i = low - 1

        for j in range(low, high):
            if arr[j] < pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]

        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    def quicksort_helper(low: int, high: int):
        if low < high:
            pivot = partition(low, high)
            quicksort_helper(low, pivot - 1)
            quicksort_helper(pivot + 1, high)

    quicksort_helper(0, len(arr) - 1)
    return arr


def heapsort(arr: list[T], inplace=True) -> list[T]:
    """
    Perform heap sort on the input list.

    Heap sort is an in-place, unstable sorting algorithm that uses a binary heap data structure.
    It divides the input into a sorted and an unsorted region, and iteratively shrinks the
    unsorted region by extracting the largest element and moving it to the sorted region.

    Args:
        arr (list[T]): The input list to be sorted.
        inplace (bool): Whether to sort the list in-place.

    Returns:
        list[T]: The sorted list.

    Time complexity:
        Best case: O(n log n) where n is the number of elements.
        Worst case: O(n log n) where n is the number of elements.
        Average case: O(n log n) where n is the number of elements.

    Space complexity:
        O(1) as sorting is done in-place.
    """
    if not inplace:
        arr = arr.copy()

    def heapify(n: int, i: int):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[left] > arr[largest]:
            largest = left

        if right < n and arr[right] > arr[largest]:
            largest = right

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(n, largest)

    n = len(arr)

    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(n, i)

    # Extract elements from the heap one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(i, 0)

    return arr


def timsort(arr: list[T], inplace=True, min_merge=32) -> list[T]:
    """
    Perform Timsort on the input list.

    Timsort is a hybrid, stable sorting algorithm that combines the strengths of
    merge sort and insertion sort. It's designed to perform well on many kinds of
    real-world data and is the default sorting algorithm in Python's sorted() and
    list.sort().

    Args:
        arr (list[T]): The input list to be sorted.
        inplace (bool): Whether to sort the list in-place.

    Returns:
        list[T]: The sorted list.

    Time complexity:
        Best case: O(n) where n is the number of elements (when the array is already sorted).
        Worst case: O(n log n) where n is the number of elements.
        Average case: O(n log n) where n is the number of elements.

    Space complexity:
        O(n) due to the merge operation.
    """
    if not inplace:
        arr = arr.copy()

    if len(arr) <= 1:
        return arr

    def calc_min_run(n):
        """Calculate the minimum length of a run."""
        r = 0
        while n >= min_merge:
            r |= n & 1
            n >>= 1
        return n + r

    def insertion_sort(arr, start, end):
        """Perform insertion sort on a small slice of the array."""
        for i in range(start + 1, end):
            key = arr[i]
            j = i - 1
            while j >= start and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

    def merge(arr, start, mid, end):
        """Merge two sorted subarrays."""
        left = arr[start:mid]
        right = arr[mid:end]
        i, j, k = 0, 0, start

        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1

        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1

    n = len(arr)
    min_run = calc_min_run(n)

    # Sort individual subarrays of size min_run
    for start in range(0, n, min_run):
        end = min(start + min_run, n)
        insertion_sort(arr, start, end)

    # Merge sorted subarrays
    size = min_run
    while size < n:
        for start in range(0, n, size * 2):
            mid = start + size
            end = min(start + size * 2, n)
            if mid < end:
                merge(arr, start, mid, end)
        size *= 2

    return arr


def counting_sort(arr: list[int], inplace=True) -> list[int]:
    """
    Perform counting sort on the input list of integers.

    Counting sort is a non-comparative sorting algorithm that operates by counting
    the number of objects that have each distinct key value, and using arithmetic on
    those counts to determine the positions of each key value in the output sequence.
    It is efficient when the range of potential items is not significantly greater
    than the number of items.

    Args:
        arr (list[int]): The input list of integers to be sorted.
        inplace (bool): Whether to sort the list in-place.

    Returns:
        list[int]: The sorted list.

    Time complexity:
        Best case: O(n + k) where n is the number of elements and k is the range of input.
        Worst case: O(n + k) where n is the number of elements and k is the range of input.
        Average case: O(n + k) where n is the number of elements and k is the range of input.

    Space complexity:
        O(k) where k is the range of input.
    """
    if not arr:
        return arr

    if not inplace:
        arr = arr.copy()

    # Find the range of the input
    max_val = max(arr)
    min_val = min(arr)
    range_val = max_val - min_val + 1

    # Initialize the counting array
    count = [0] * range_val

    # Count the occurrences of each element
    for num in arr:
        count[num - min_val] += 1

    # Modify count array to store actual position of elements in output array
    for i in range(1, len(count)):
        count[i] += count[i - 1]

    # Build the output array
    n = len(arr)
    output = [0] * n
    for i in range(n - 1, -1, -1):
        output[count[arr[i] - min_val] - 1] = arr[i]
        count[arr[i] - min_val] -= 1

    # Copy the output array to arr if sorting in-place
    for i in range(n):
        arr[i] = output[i]

    return arr


def radix_sort(arr: list[int], inplace=True, radix=10) -> list[int]:
    """
    Perform radix sort on the input list of non-negative integers.

    Radix sort is a non-comparative integer sorting algorithm that sorts data
    with integer keys by grouping keys by the individual digits which share the
    same significant position and value. It sorts the elements by processing each
    digit or character position, starting from the least significant digit to the
    most significant digit.

    Args:
        arr (list[int]): The input list of non-negative integers to be sorted.
        inplace (bool): Whether to sort the list in-place.
        radix (int): The base of the number system (typically 10 for decimal).

    Returns:
        list[int]: The sorted list.

    Time complexity:
        Best case: O(d * (n + k)) where n is the number of elements,
                   k is the range of each digit (10 for decimal),
                   and d is the number of digits in the maximum element.
        Worst case: O(d * (n + k))
        Average case: O(d * (n + k))

    Space complexity:
        O(n + k) where n is the number of elements and k is the range of each digit (typically 10).

    Note:
        This implementation assumes non-negative integers. For negative numbers,
        the algorithm needs to be modified.
    """
    if not arr:
        return arr

    if not inplace:
        arr = arr.copy()

    def counting_sort_for_digit(arr: list[int], exp: int) -> None:
        n = len(arr)
        output = [0] * n
        count = [0] * radix

        # Store count of occurrences in count[]
        for i in range(n):
            index = arr[i] // exp
            count[index % radix] += 1

        # Change count[i] so that count[i] now contains actual
        # position of this digit in output[]
        for i in range(1, radix):
            count[i] += count[i - 1]

        # Build the output array
        i = n - 1
        while i >= 0:
            index = arr[i] // exp
            output[count[index % radix] - 1] = arr[i]
            count[index % radix] -= 1
            i -= 1

        # Copy the output array to arr[], so that arr[] now
        # contains sorted numbers according to current digit
        for i in range(n):
            arr[i] = output[i]

    # Find the maximum number to know number of digits
    max_num = max(arr)

    # Do counting sort for every digit
    exp = 1
    while max_num // exp > 0:
        counting_sort_for_digit(arr, exp)
        exp *= radix

    return arr


def bucket_sort(
    arr: list[int | float], inplace=True, n_buckets=10, sort_fn=insertion_sort
) -> list[int | float]:
    """
    Perform bucket sort on the input list.

    Bucket sort is a distribution-based sorting algorithm that works by distributing the
    elements into a number of buckets, then sorting these buckets individually. It works
    best when the input is uniformly distributed over a range.

    Args:
        arr (list[int|float]): The input list to be sorted.
        inplace (bool): Whether to sort the list in-place.
        n_buckets (int): The number of buckets to use.
        sort_fn (Callable): The sorting function to use for sorting the individual buckets.

    Returns:
        list[int|float]: The sorted list.

    Time complexity:
        Best case: O(n + k) where n is the number of elements and k is the number of buckets.
        Worst case: O(n^2) when most elements are in the same bucket.
        Average case: O(n + k) when elements are uniformly distributed.

    Space complexity:
        O(n + k) where n is the number of elements and k is the number of buckets.
    """
    if not inplace:
        arr = arr.copy()

    n = len(arr)
    if n <= 1:
        return arr

    # Create buckets
    buckets = [[] for _ in range(n_buckets)]
    min_val, max_val = min(arr), max(arr)

    # Distribute elements into buckets
    for num in arr:
        index = int((num - min_val) / (max_val - min_val) * (n_buckets - 1))
        buckets[index].append(num)

    # Sort individual buckets (using insertion sort for simplicity)
    for bucket in buckets:
        sort_fn(bucket, inplace=True)

    # Concatenate sorted buckets
    sorted_arr = [num for bucket in buckets for num in bucket]

    # Copy sorted elements back to original array if sorting in-place
    if inplace:
        arr[:] = sorted_arr
    else:
        arr = sorted_arr

    return arr


# Benchmarking functions

fn_by_name = {
    "bogo_sort": bogo_sort,
    "bubble_sort": bubble_sort,
    "comb_sort": comb_sort,
    "cycle_sort": cycle_sort,
    "selection_sort": selection_sort,
    "insertion_sort": insertion_sort,
    "gnome_sort": gnome_sort,
    "shell_sort": shell_sort,
    "quicksort": quicksort,
    "heapsort": heapsort,
    "merge_sort": merge_sort,
    "timsort": timsort,
    "bucket_sort": bucket_sort,
    "radix_sort": radix_sort,
    "counting_sort": counting_sort,
}


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        "-t",
        type=str,
        nargs="*",
        default=[fn for fn in fn_by_name],
        help="Algorithms to include in testing",
    )

    parser.add_argument(
        "--criteria",
        "-c",
        type=str,
        default="time",
        help="Criteria to benchmark (time, memory)",
    )

    parser.add_argument(
        "--start",
        "-s",
        type=int,
        default=0,
        help="Start list size",
    )

    parser.add_argument(
        "--end",
        "-e",
        type=int,
        default=1000,
        help="End list size",
    )

    parser.add_argument(
        "--resolution",
        "-r",
        type=int,
        default=20,
        help="Number of list sizes to test between min and max",
    )

    parser.add_argument(
        "--n_samples",
        "-n",
        type=int,
        default=100,
        help="Number of samples per list size (over which a statistic will be taken)",
    )

    parser.add_argument(
        "--metric",
        "-m",
        type=str,
        default="median",
        help="Statistic applied to samples (mean, median, max, min)",
    )

    parser.add_argument(
        "--distribution",
        "-d",
        type=str,
        default="randint",
        help="Probability distribution to use for list numbers (uniform, randint, poisson, etc.)",
    )

    parser.add_argument(
        "--params",
        "-p",
        type=float,
        nargs="*",
        default=[0, 1000],
        help="Parameters for the probability distribution",
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


def func_time(fn: Callable, *args, **kwargs) -> float:
    start = time.perf_counter()
    fn(*args, **kwargs)
    end = time.perf_counter()
    return end - start


def func_memory(fn: Callable, *args, **kwargs) -> float:
    tracemalloc.start()
    fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return float(peak)


def run_single_sample(
    list_sz: int,
    criteria: str,
    fn_by_name: dict[str, Callable],
    rand_fn: Callable[[], float],
) -> dict[str, float]:
    lst = [rand_fn() for _ in range(list_sz)]
    measurement_by_fn = {}
    for name, fn in fn_by_name.items():
        cpy = lst.copy()
        if criteria == "time":
            measurement_by_fn[name] = func_time(fn, cpy)
        if criteria == "memory":
            measurement_by_fn[name] = func_memory(fn, cpy)
    return measurement_by_fn


def benchmark_listsz(
    list_sz: int,
    criteria: str,
    fn_by_name: dict[str, Callable],
    n_samples: int,
    metric_fn: Callable[[list[float]], float],
    rand_fn: Callable[[], float],
) -> dict[str, float]:
    with mp.Pool() as pool:
        run_sample = partial(
            run_single_sample,
            criteria=criteria,
            fn_by_name=fn_by_name,
            rand_fn=rand_fn,
        )
        samples = pool.map(run_sample, [list_sz] * n_samples)

    metric_by_fn = {}
    for name in fn_by_name:
        measurements = [sample[name] for sample in samples]
        metric_by_fn[name] = metric_fn(measurements)

    return metric_by_fn


def benchmark_functions(
    fn_by_name: dict[str, Callable],
    n_samples: int,
    list_sizes: list[int],
    rand_fn: Callable[..., float],
    criteria: str,
    metric_fn: Callable[[list[float]], float],
) -> dict[str, list[float]]:
    metrics_by_fn = {name: [0.0] * len(list_sizes) for name in fn_by_name}

    for i, list_sz in enumerate(list_sizes):
        metric_by_fn = benchmark_listsz(
            list_sz, criteria, fn_by_name, n_samples, metric_fn, rand_fn
        )
        for name, metric in metric_by_fn.items():
            metrics = metrics_by_fn[name]
            metrics[i] = metric
        print(
            f"\33[2K\rCompleted {i+1}/{len(list_sizes)} list sizes (size {list_sz}/{max(list_sizes)})",
            end="",
        )

    return metrics_by_fn


def plot_benchmarks(
    list_sizes: list[int],
    metrics_by_fn: dict[str, list[float]],
    criteria_name: str,
    metric_name: str,
) -> None:
    for name, metrics in metrics_by_fn.items():
        plt.plot(list_sizes, metrics, label=name)
    unit = "seconds" if criteria_name == "time" else "bytes"
    plt.ylabel(f"{metric_name} {criteria_name} ({unit})")
    plt.xlabel("list size")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    # get the script arguments
    args = get_args()
    args.params = [int(param) if param.is_integer() else param for param in args.params]

    # print the arguments
    print("Running with args:")
    for arg, val in vars(args).items():
        if isinstance(val, list):
            prefix = len(f"    {arg}: ")
            separator = ", \n" + " " * prefix
            val = separator.join(str(element) for element in val)
        print(f"    {arg}: {val}")

    # collect the functions to test
    fns = {
        name: fn
        for name, fn in fn_by_name.items()
        if name in args.test and name not in args.blacklist
    }

    # setup the list sizes to test on
    step = (args.end - args.start) // args.resolution
    list_sizes = [
        round(i * (args.end - args.start) / args.resolution + args.start)
        for i in range(args.resolution + 1)
    ]

    # setup statistical tools
    distribution = getattr(stats, args.distribution)(*args.params)
    rand_fn = distribution.rvs
    metric_fn = getattr(numpy, args.metric)

    print("\nBenchmarking functions")
    metrics_by_fn = benchmark_functions(
        fns, args.n_samples, list_sizes, rand_fn, args.criteria, metric_fn
    )
    print()
    plot_benchmarks(list_sizes, metrics_by_fn, args.criteria, args.metric)
