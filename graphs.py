class Node:
    def __init__(self, key):
        self.key = key
        self.neighbors = set()

    def add_neighbor(self, neighbor):
        self.neighbors.add(neighbor)

    def __str__(self):
        return str(self.key)

    def __repr__(self):
        return f"Node({self.key}, {[str(n) for n in self.neighbors])})"


class Graph:
    def __init__(self):
        self.nodes_by_key = {}

    def add_node(self, key):
        self.nodes_by_key[key] = Node(key)

    def add_edge(self, key1, key2):
        node1 = self.nodes_by_key[key1]
        node2 = self.nodes_by_key[key2]
        node1.add_neighbor(node2)
        node2.add_neighbor(node1)

    def __str__(self):
        return str(list(self.nodes_by_key.keys()))

    def __repr__(self):
        return "\n".join([repr(node) for node in self.nodes_by_key.values()])

    def __getitem__(self, key):
        return self.nodes_by_key[key]

    def __contains__(self, key):
        return key in self.nodes_by_key

    def __iter__(self):
        return iter(self.nodes_by_key.values())

    def __len__(self):
        return len(self.nodes_by_key)

    def __bool__(self):
        return bool(self.nodes_by_key)


class WeightedNode:
    def __init__(self, key):
        self.key = key
        self.neighbors = {}

    def add_neighbor(self, neighbor, weight):
        self.neighbors[neighbor] = weight

    def __str__(self):
        return str(self.key)

    def __repr__(self):
        return f"WeightedNode({self.key}, {[f"{str(n)} ({w})" for n, w in self.neighbors]})"


class WeightedGraph:
    def __init__(self):
        self.nodes_by_key = {}

    def add_node(self, key):
        self.nodes_by_key[key] = WeightedNode(key)

    def add_edge(self, key1, key2, weight):
        node1 = self.nodes_by_key[key1]
        node2 = self.nodes_by_key[key2]
        node1.add_neighbor(node2, weight)
        node2.add_neighbor(node1, weight)

    def __str__(self):
        return str(list(self.nodes_by_key.keys()))

    def __repr__(self):
        return "\n".join([repr(node) for node in self.nodes_by_key.values()])

    def __getitem__(self, key):
        return self.nodes_by_key[key]

    def __contains__(self, key):
        return key in self.nodes_by_key

    def __iter__(self):
        return iter(self.nodes_by_key.values())

    def __len__(self):
        return len(self.nodes_by_key)

    def __bool__(self):
        return bool(self.nodes_by_key)


