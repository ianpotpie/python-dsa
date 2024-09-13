class Array:
    def __init__(self, size):
        self.values: list = size * [0]

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        if type(idx) != "int":
            raise TypeError("index must be an integer")

        if not (-len(self) <= idx < len(self)):
            raise IndexError("index out of range")

        return self.values[idx]

    def __setitem__(self, idx, val):
        if type(idx) != "int":
            raise TypeError("index must be an integer")

        if not (-len(self) <= idx < len(self)):
            raise IndexError("index out of range")

        self.values[idx] = val


class ArrayList:
    def __init__(self, init_size=16, resize_ratio=1.5):
        self.size: int = 0
        self.resize_ratio: float = resize_ratio
        self.array: Array = Array(init_size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if type(idx) != "int":
            raise TypeError("index must be an integer")

        if not (-len(self) <= idx < len(self)):
            raise IndexError("index out of range")

        return self.array[idx]

    def __setitem__(self, idx, val):
        if type(idx) != "int":
            raise TypeError("index must be an integer")

        if not (-len(self) <= idx < len(self)):
            raise IndexError("index out of range")

        self.array[idx] = val

    def __str__(self):
        s = "["
        for i in range(self.size):
            s += str(self.array[i])
            if i < self.size - 1:
                s += ", "
        s += "]"
        return s

    def __repr__(self):
        return str(self)

    def __contains__(self, val):
        for i in range(self.size):
            if self.array[i] == val:
                return True
        return False

    def __eq__(self, other):
        if len(self) != len(other):
            return False

        for i in range(len(self)):
            if self[i] != other[i]:
                return False
        return True

    def __iter__(self):
        for i in range(self.size):
            yield self.array[i]

    def __bool__(self):
        return bool(self.size)

    def __add__(self, other):
        new_array = ArrayList(len(self) + len(other))
        for val in self:
            new_array.append(val)
        for val in other:
            new_array.append(val)
        return new_array

    def expand(self):
        new_array = Array(int(len(self) * self.resize_ratio) + 1)
        for i in range(len(self)):
            new_array[i] = self.array[i]
        self.array = new_array

    def shrink(self):
        new_array = Array(int(len(self) / self.resize_ratio) + 1)
        for i in range(len(self)):
            new_array[i] = self.array[i]
        self.array = new_array

    def append(self, val):
        if self.size == len(self.array):
            self.expand()
        self.array[self.size] = val
        self.size += 1

    def insert(self, idx, val):
        if self.size == len(self.array):
            self.expand()
        for i in range(self.size, idx, -1):
            self.array[i] = self.array[i - 1]
        self.array[idx] = val
        self.size += 1

    def remove(self, val):
        for i in range(self.size):
            if self.array[i] == val:
                for j in range(i, self.size - 1):
                    self.array[j] = self.array[j + 1]
                self.size -= 1

                if self.size < len(self.array) / (self.resize_ratio**2):
                    self.shrink()

                return

        raise ValueError("value not found")

    def pop(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index out of range")

        val = self.array[idx]
        for i in range(idx, self.size - 1):
            self.array[i] = self.array[i + 1]
        self.size -= 1

        if self.size < len(self.array) / (self.resize_ratio**2):
            self.shrink()

        return val

    def clear(self):
        self.size = 0

    def index(self, val):
        for i in range(self.size):
            if self.array[i] == val:
                return i
        raise ValueError("value not found")

    def count(self, val):
        count = 0
        for i in range(self.size):
            if self.array[i] == val:
                count += 1
        return count

    def reverse(self):
        for i in range(self.size // 2):
            self.array[i], self.array[self.size - i - 1] = (
                self.array[self.size - i - 1],
                self.array[i],
            )

    def sort(self):
        # uses quicksort
        def partition(low, high):
            pivot = self.array[high]
            i = low - 1
            for j in range(low, high):
                if self.array[j] < pivot:
                    i += 1
                    self.array[i], self.array[j] = self.array[j], self.array[i]
            self.array[i + 1], self.array[high] = self.array[high], self.array[i + 1]
            return i + 1

        def quicksort(low, high):
            if low < high:
                pi = partition(low, high)
                quicksort(low, pi - 1)
                quicksort(pi + 1, high)

        quicksort(0, self.size - 1)

    def extend(self, other):
        for val in other:
            self.append(val)

    def copy(self):
        new_array = ArrayList(len(self))
        for val in self:
            new_array.append(val)
        return new_array


class SinglyLinkedNode:
    def __init__(self, value):
        self.value = value
        self.next: SinglyLinkedNode | None = None


class SinglyLinkedList:
    def __init__(self):
        self.head: SinglyLinkedNode | None = None
        self.size: int = 0

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if type(idx) != "int":
            raise TypeError("index must be an integer")

        if idx < 0:
            idx += len(self)

        current = self.head
        for _ in range(idx):
            if not current:
                raise IndexError("index out of range")
            current = current.next

        if not current:
            raise IndexError("index out of range")
        return current.value

    def __setitem__(self, idx, val):
        if type(idx) != "int":
            raise TypeError("index must be an integer")

        if idx < 0:
            idx += len(self)

        current = self.head
        for _ in range(idx):
            if not current:
                raise IndexError("index out of range")
            current = current.next

        if not current:
            raise IndexError("index out of range")
        current.value = val

    def __str__(self):
        s = "["
        current = self.head
        while current:
            s += str(current.value)
            if current.next:
                s += ", "
            current = current.next
        s += "]"
        return s

    def __repr__(self):
        return str(self)

    def __contains__(self, val):
        current = self.head
        while current:
            if current.value == val:
                return True
            current = current.next
        return False

    def __eq__(self, other):
        if len(self) != len(other):
            return False

        current1 = self.head
        current2 = other.head
        while current1:
            if current1.value != current2.value:
                return False
            current1 = current1.next
            current2 = current2.next
        return True

    def __iter__(self):
        current = self.head
        while current:
            yield current.value
            current = current.next

    def __bool__(self):
        return bool(self.size)

    def append(self, val):
        if not self.head:
            self.head = SinglyLinkedNode(val)
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = SinglyLinkedNode(val)
        self.size += 1

    def insert(self, idx, val):
        if type(idx) != "int":
            raise TypeError("index must be an integer")

        if idx < 0:
            idx += len(self) + 1

        if idx == 0:
            new_node = SinglyLinkedNode(val)
            new_node.next = self.head
            self.head = new_node
            self.size += 1
            return

        current = self.head
        for _ in range(idx - 1):
            if not current:
                raise IndexError("index out of range")
            current = current.next

        if not current:
            raise IndexError("index out of range")

        new_node = SinglyLinkedNode(val)
        new_node.next = current.next
        current.next = new_node
        self.size += 1

    def remove(self, val):
        if not self.head:
            raise ValueError("value not found")

        if self.head.value == val:
            self.head = self.head.next
            self.size -= 1
            return

        current = self.head
        while current.next:
            if current.next.value == val:
                current.next = current.next.next
                self.size -= 1
                return
            current = current.next

        raise ValueError("value not found")

    def pop(self, idx):
        if type(idx) != "int":
            raise TypeError("index must be an integer")

        if idx < 0:
            idx += len(self)

        if idx == 0:
            if not self.head:
                raise IndexError("index out of range")
            val = self.head.value
            self.head = self.head.next
            self.size -= 1
            return val

        current = self.head
        for _ in range(idx - 1):
            if not current:
                raise IndexError("index out of range")
            current = current.next

        if not current or not current.next:
            raise IndexError("index out of range")

        val = current.next.value
        current.next = current.next.next
        self.size -= 1
        return val

    def clear(self):
        self.head = None
        self.size = 0

    def index(self, val):
        current = self.head
        idx = 0
        while current:
            if current.value == val:
                return idx
            current = current.next
            idx += 1
        raise ValueError("value not found")

    def count(self, val):
        count = 0
        current = self.head
        while current:
            if current.value == val:
                count += 1
            current = current.next
        return count

    def reverse(self):
        current = self.head
        prev = None
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        self.head = prev

    def sort(self):
        # uses selection sort
        current = self.head
        while current:
            min_node = current
            next_node = current.next
            while next_node:
                if next_node.value < min_node.value:
                    min_node = next_node
                next_node = next_node.next
            current.value, min_node.value = min_node.value, current.value
            current = current.next

    def extend(self, other):
        if not self.head:
            self.head = other.head
            self.size = len(other)
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = other.head
        self.size += len(other)

    def copy(self):
        new_list = SinglyLinkedList()
        current = self.head
        while current:
            new_list.append(current.value)
            current = current.next
        return new_list


class DoublyLinkedNode:
    def __init__(self, value):
        self.value = value
        self.next: DoublyLinkedNode | None = None
        self.prev: DoublyLinkedNode | None = None


class DoublyLinkedList:
    def __init__(self):
        self.head: DoublyLinkedNode | None = None
        self.tail: DoublyLinkedNode | None = None
        self.size: int = 0

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if type(idx) != "int":
            raise TypeError("index must be an integer")

        current = self.head
        for _ in range(idx):
            if not current:
                raise IndexError("index out of range")
            current = current.next if idx > 0 else current.prev

        if not current:
            raise IndexError("index out of range")
        return current.value

    def __setitem__(self, idx, val):
        if type(idx) != "int":
            raise TypeError("index must be an integer")

        current = self.head
        for _ in range(idx):
            if not current:
                raise IndexError("index out of range")
            current = current.next if idx > 0 else current.prev

        if not current:
            raise IndexError("index out of range")
        current.value = val

    def __str__(self):
        s = "["
        current = self.head
        while current:
            s += str(current.value)
            if current.next:
                s += ", "
            current = current.next
        s += "]"
        return s

    def __repr__(self):
        return str(self)

    def __contains__(self, val):
        current = self.head
        while current:
            if current.value == val:
                return True
            current = current.next
        return False

    def __eq__(self, other):
        if len(self) != len(other):
            return False

        current1 = self.head
        current2 = other.head
        while current1:
            if current1.value != current2.value:
                return False
            current1 = current1.next
            current2 = current2.next
        return True

    def __iter__(self):
        current = self.head
        while current:
            yield current.value
            current = current.next

    def __bool__(self):
        return bool(self.size)

    def append(self, val):
        new_node = DoublyLinkedNode(val)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.prev = self.tail
            if not self.tail:
                raise ValueError("tail is None")
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1

    def insert(self, idx, val):
        if type(idx) != "int":
            raise TypeError("index must be an integer")

        if idx == 0:
            new_node = DoublyLinkedNode(val)
            new_node.next = self.head
            if not self.head:
                raise ValueError("head is None")
            self.head.prev = new_node
            self.head = new_node
            self.size += 1
            return

        if idx == -1 or idx == len(self) - 1:
            self.append(val)
            return

        current = self.head if idx > 0 else self.tail
        for _ in range(abs(idx) - 1):
            if not current:
                raise IndexError("index out of range")
            current = current.next if idx > 0 else current.prev

        if not current:
            raise IndexError("index out of range")

        new_node = DoublyLinkedNode(val)
        new_node.prev = current
        new_node.next = current.next if idx > 0 else current.prev
        if idx > 0:
            current.next = new_node
        else:
            current.prev = new_node
        self.size += 1

    def remove(self, val):
        if not self.head:
            raise ValueError("value not found")

        if self.head.value == val:
            self.head = self.head.next
            if self.head:
                self.head.prev = None
            self.size -= 1
            return

        current = self.head
        while current.next:
            if current.next.value == val:
                current.next = current.next.next
                if current.next:
                    current.next.prev = current
                self.size -= 1
                return
            current = current.next

        raise ValueError("value not found")

    def pop(self, idx):
        if type(idx) != "int":
            raise TypeError("index must be an integer")

        if idx == 0:
            if not self.head:
                raise IndexError("index out of range")
            val = self.head.value
            self.head = self.head.next
            if self.head:
                self.head.prev = None
            self.size -= 1
            return val

        if idx == -1 or idx == len(self) - 1:
            if not self.tail:
                raise ValueError("tail is None")
            val = self.tail.value
            self.tail = self.tail.prev
            if self.tail:
                self.tail.next = None
            self.size -= 1
            return val

        current = self.head if idx > 0 else self.tail
        for _ in range(abs(idx) - 1):
            if not current:
                raise IndexError("index out of range")
            current = current.next if idx > 0 else current.prev

        if not current:
            raise IndexError("index out of range")

        if not current.next:
            raise IndexError("index out of range")
        val = current.next.value
        current.next = current.next.next
        if current.next:
            current.next.prev = current
        self.size -= 1
        return val

    def clear(self):
        self.head = None
        self.tail = None
        self.size = 0

    def index(self, val):
        current = self.head
        idx = 0
        while current:
            if current.value == val:
                return idx
            current = current.next
            idx += 1
        raise ValueError("value not found")

    def count(self, val):
        count = 0
        current = self.head
        while current:
            if current.value == val:
                count += 1
            current = current.next
        return count

    def reverse(self):
        current = self.head
        while current:
            current.prev, current.next = current.next, current.prev
            current = current.prev
        self.head, self.tail = self.tail, self.head

    def sort(self):
        # uses selection sort
        current = self.head
        while current:
            min_node = current
            next_node = current.next
            while next_node:
                if next_node.value < min_node.value:
                    min_node = next_node
                next_node = next_node.next
            current.value, min_node.value = min_node.value, current.value
            current = current.next

    def extend(self, other):
        if not self.head:
            self.head = other.head
            self.tail = other.tail
            self.size = len(other)
            return

        if not self.tail:
            raise ValueError("tail is None")
        self.tail.next = other.head
        other.head.prev = self.tail
        self.tail = other.tail
        self.size += len(other)

    def copy(self):
        new_list = DoublyLinkedList()
        current = self.head
        while current:
            new_list.append(current.value)
            current = current.next
        return new_list


class Queue:
    def __init__(self):
        self.queue = DoublyLinkedList()

    def __len__(self):
        return len(self.queue)

    def __str__(self):
        return str(self.queue)

    def __repr__(self):
        return str(self)

    def __contains__(self, val):
        return val in self.queue

    def __eq__(self, other):
        return self.queue == other.queue

    def __iter__(self):
        return iter(self.queue)

    def __bool__(self):
        return bool(self.queue)

    def enqueue(self, val):
        self.queue.append(val)

    def dequeue(self):
        return self.queue.pop(0)

    def clear(self):
        self.queue.clear()


class Stack:
    def __init__(self):
        self.stack = SinglyLinkedList()

    def __len__(self):
        return len(self.stack)

    def __str__(self):
        return str(self.stack)

    def __repr__(self):
        return str(self)

    def __contains__(self, val):
        return val in self.stack

    def __eq__(self, other):
        return self.stack == other.stack

    def __iter__(self):
        return iter(self.stack)

    def __bool__(self):
        return bool(self.stack)

    def push(self, val):
        self.stack.append(val)

    def pop(self):
        return self.stack.pop(-1)

    def clear(self):
        self.stack.clear()
