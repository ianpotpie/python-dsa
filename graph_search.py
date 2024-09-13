import heapq

from graphs import Node


def depth_first_search(start, end):
    node_to_prev = {start: None}
    stack = [start]

    while stack:
        current = stack.pop()
        if current is end:
            break

        for neighbor in current.neighbors:
            if neighbor not in node_to_prev:
                stack.append(neighbor)
                node_to_prev[neighbor] = current

    path = [end]
    while path[-1] is not start:
        prev = node_to_prev[path[-1]]
        path = [prev] + path

    return path

def dfs_with_revisiting(start, end):
    node_to_prev = {start: None}
    node_to_dist = {start: 0}
    stack = [(start, 0)]

    while stack:
        current, depth = stack.pop()
        if current is end:
            break

        for neighbor in current.neighbors:
            neighbor_unvisited = neighbor not in node_to_prev
            neighbor_closer = node_to_dist.get(neighbor, 0) > depth + 1
            if neighbor_unvisited or neighbor_closer:
                stack.append((neighbor, depth + 1))
                node_to_prev[neighbor] = current
                node_to_dist[neighbor] = depth + 1

    path = [end]
    while path[-1] is not start:
        prev = node_to_prev[path[-1]]
        path = [prev] + path

    return path

def recursive_dfs(start, end, visited=None):
    if start is end:
        return [end] 

    if visited is None:
        visited = set()

    for neighbor in start.neighbors:
        if neighbor not in visited:
            path = recursive_dfs(neighbor, end)
            if path:
                return [start] + path

    return []


def breadth_first_search(start, end=None):
    node_to_prev = {start: None}
    queue = [start]

    while queue:
        current = queue.pop(0)
        if current is end:
            break

        for neighbor in current.neighbors:
            if neighbor not in node_to_prev:
                queue.append(neighbor)
                node_to_prev[neighbor] = current

    path = [end]
    while path[-1] is not start:
        prev = node_to_prev[path[-1]]
        path = [prev] + path

    return path


def depth_limited_search(start, depth, end, limit):
    if start is end:
        return [end]

    if depth == limit:
        return None

    for neighbor in start.neighbors:
        path = depth_limited_search(neighbor, depth + 1, end, limit)
        if path is not None:
            return [start] + path

    return None


def iterative_deepening_search(start, end):
    limit = 0
    while True:
        path = depth_limited_search(start, 0, end, limit)
        if path is not None:
            return path
        limit += 1


def uniform_cost_search(start, end):
    node_to_prev = {start: None}
    node_to_cost = {start: 0}
    queue = [(0, start)]

    while queue:
        cost, current = heapq.heappop(queue)
        if current is end:
            break

        for neighbor, weight in current.neighbors.items():
            new_cost = cost + weight
            if neighbor not in node_to_cost or new_cost < node_to_cost[neighbor]:
                node_to_cost[neighbor] = new_cost
                heapq.heappush(queue, (new_cost, neighbor))
                node_to_prev[neighbor] = current

    path = [end]
    while path[-1] is not start:
        prev = node_to_prev[path[-1]]
        path = [prev] + path

    return path


def greedy_best_first_search(start, end, heuristic):
    node_to_prev = {start: None}
    queue = [(heuristic(start, end), start)]

    while queue:
        _, current = heapq.heappop(queue)
        if current is end:
            break

        for neighbor in current.neighbors:
            if neighbor not in node_to_prev:
                heapq.heappush(queue, (heuristic(neighbor, end), neighbor))
                node_to_prev[neighbor] = current

    path = [end]
    while path[-1] is not start:
        prev = node_to_prev[path[-1]]
        path = [prev] + path

    return path


def a_star_search(start, end, heuristic):
    node_to_prev = {start: None}
    node_to_cost = {start: 0}
    queue = [(heuristic(start, end), 0, start)]

    while queue:
        _, cost, current = heapq.heappop(queue)
        if current is end:
            break

        for neighbor, weight in current.neighbors.items():
            new_cost = node_to_cost[current] + weight
            if neighbor not in node_to_cost or new_cost < node_to_cost[neighbor]:
                node_to_cost[neighbor] = new_cost
                heapq.heappush(queue, (new_cost + heuristic(neighbor, end), new_cost, neighbor))
                node_to_prev[neighbor] = current

    path = [end]
    while path[-1] is not start:
        prev = node_to_prev[path[-1]]
        path = [prev] + path

    return path


def bidirectional_dfs(start, end):
    start_to_prev = {start: None}
    end_to_prev = {end: None}
    start_stack = [start]
    end_stack = [end]

    while start_stack and end_stack:
        start_current = start_stack.pop()
        end_current = end_stack.pop()

        if start_current in end_to_prev or end_current in start_to_prev:
            start_path = [start_current] if start_current in end_to_prev else [end_current]
            while start_path[-1] is not start:
                prev = start_to_prev[start_path[-1]]
                start_path = [prev] + start_path 

            end_path = [start_current] if start_current in end_to_prev else [end_current]
            while end_path[-1] is not end:
                prev = end_to_prev[end_path[-1]]
                end_path = [prev] + end_path

            return start_path + end_path[::-1]

        for start_neighbor in start_current.neighbors:
            if start_neighbor not in start_to_prev:
                start_stack.append(start_neighbor)
                start_to_prev[start_neighbor] = start_current

        for end_neighbor in end_current.neighbors:
            if end_neighbor not in end_to_prev:
                end_stack.append(end_neighbor)
                end_to_prev[end_neighbor] = end_current

    return None


def bidirectional_bfs(start, end):
    start_to_prev = {start: None}
    end_to_prev = {end: None}
    start_queue = [start]
    end_queue = [end]

    while start_queue and end_queue:
        start_current = start_queue.pop(0)
        end_current = end_queue.pop(0)

        if start_current in end_to_prev or end_current in start_to_prev:
            start_path = [start_current] if start_current in end_to_prev else [end_current]
            while start_path[-1] is not start:
                prev = start_to_prev[start_path[-1]]
                start_path = [prev] + start_path

            end_path = [start_current] if start_current in end_to_prev else [end_current]
            while end_path[-1] is not end:
                prev = end_to_prev[end_path[-1]]
                end_path = [prev] + end_path

            return start_path + end_path[::-1]

        for start_neighbor in start_current.neighbors:
            if start_neighbor not in start_to_prev:
                start_queue.append(start_neighbor)
                start_to_prev[start_neighbor] = start_current

        for end_neighbor in end_current.neighbors:
            if end_neighbor not in end_to_prev:
                end_queue.append(end_neighbor)
                end_to_prev[end_neighbor] = end_current

    return None


def bidirectional_ucs(start, end):
    start_to_prev = {start: None}
    end_to_prev = {end: None}
    start_to_cost = {start: 0}
    end_to_cost = {end: 0}
    start_queue = [(0, start)]
    end_queue = [(0, end]

    while start_queue and end_queue:
        start_cost, start_current = heapq.heappop(start_queue)
        end_cost, end_current = heapq.heappop(end_queue)

        if start_current in end_to_prev or end_current in start_to_prev:
            start_path = [start_current] if start_current in end_to_prev else [end_current]
            while start_path[-1] is not start:
                prev = start_to_prev[start_path[-1]]
                start_path = [prev] + start_path

            end_path = [start_current] if start_current in end_to_prev else [end_current]
            while end_path[-1] is not end:
                prev = end_to_prev[end_path[-1]]
                end_path = [prev] + end_path

            return start_path + end_path[::-1]

        for start_neighbor, weight in start_current.neighbors.items():
            new_cost = start_to_cost[start_current] + weight
            if start_neighbor not in start_to_cost or new_cost < start_to_cost[start_neighbor]:
                start_to_cost[start_neighbor] = new_cost
                heapq.heappush(start_queue, (new_cost, start_neighbor))
                start_to_prev[start_neighbor] = start_current

        for end_neighbor, weight in end_current.neighbors.items():
            new_cost = end_to_cost[end_current] + weight
            if end_neighbor not in end_to_cost or new_cost < end_to_cost[end_neighbor]:
                end_to_cost[end_neighbor] = new_cost
                heapq.heappush(end_queue, (new_cost, end_neighbor))
                end_to_prev[end_neighbor] = end_current

    return None


def bidirectional_gbfs(start, end, heuristic):
    start_to_prev = {start: None}
    end_to_prev = {end: None}
    start_queue = [(heuristic(start, end), start)]
    end_queue = [(heuristic(end, start), end)]

    while start_queue and end_queue:
        _, start_current = heapq.heappop(start_queue)
        _, end_current = heapq.heappop(end_queue)

        if start_current in end_to_prev or end_current in start_to_prev:
            start_path = [start_current] if start_current in end_to_prev else [end_current]
            while start_path[-1] is not start:
                prev = start_to_prev[start_path[-1]]
                start_path = [prev] + start_path

            end_path = [start_current] if start_current in end_to_prev else [end_current]
            while end_path[-1] is not end:
                prev = end_to_prev[end_path[-1]]
                end_path = [prev] + end_path

            return start_path + end_path[::-1]

        for start_neighbor in start_current.neighbors:
            if start_neighbor not in start_to_prev:
                heapq.heappush(start_queue, (heuristic(start_neighbor, end), start_neighbor))
                start_to_prev[start_neighbor] = start_current

        for end_neighbor in end_current.neighbors:
            if end_neighbor not in end_to_prev:
                heapq.heappush(end_queue, (heuristic(end_neighbor, start), end_neighbor))
                end_to_prev[end_neighbor] = end_current

    return None


def bidirectional_astar(start, end, heuristic):
    start_to_prev = {start: None}
    end_to_prev = {end: None}
    start_to_cost = {start: 0}
    end_to_cost = {end: 0}
    start_queue = [(heuristic(start, end), 0, start)]
    end_queue = [(heuristic(end, start), 0, end)]

    while start_queue and end_queue:
        _, start_cost, start_current = heapq.heappop(start_queue)
        _, end_cost, end_current = heapq.heappop(end_queue)

        if start_current in end_to_prev or end_current in start_to_prev:
            start_path = [start_current] if start_current in end_to_prev else [end_current]
            while start_path[-1] is not start:
                prev = start_to_prev[start_path[-1]]
                start_path = [prev] + start_path

            end_path = [start_current] if start_current in end_to_prev else [end_current]
            while end_path[-1] is not end:
                prev = end_to_prev[end_path[-1]]
                end_path = [prev] + end_path

            return start_path + end_path[::-1]

        for start_neighbor, weight in start_current.neighbors.items():
            new_cost = start_to_cost[start_current] + weight
            if start_neighbor not in start_to_cost or new_cost < start_to_cost[start_neighbor]:
                start_to_cost[start_neighbor] = new_cost
                heapq.heappush(start_queue, (new_cost + heuristic(start_neighbor, end), new_cost, start_neighbor))
                start_to_prev[start_neighbor] = start_current

        for end_neighbor, weight in end_current.neighbors.items():
            new_cost = end_to_cost[end_current] + weight
            if end_neighbor not in end_to_cost or new_cost < end_to_cost[end_neighbor]:
                end_to_cost[end_neighbor] = new_cost
                heapq.heappush(end_queue, (new_cost + heuristic(end_neighbor, start), new_cost, end_neighbor))
                end_to_prev[end_neighbor] = end_current

    return None
