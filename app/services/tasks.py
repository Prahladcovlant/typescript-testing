from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Tuple


def schedule_tasks(tasks: List[str], dependencies: List[List[str]]) -> Tuple[List[str], List[str]]:
    graph: Dict[str, List[str]] = defaultdict(list)
    indegree = {task: 0 for task in tasks}

    for prerequisite, dependent in dependencies:
        graph[prerequisite].append(dependent)
        indegree[dependent] = indegree.get(dependent, 0) + 1
        indegree.setdefault(prerequisite, 0)

    queue = deque(sorted([task for task, deg in indegree.items() if deg == 0]))
    order = []

    while queue:
        task = queue.popleft()
        order.append(task)
        for neighbor in sorted(graph[task]):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(indegree):
        raise ValueError("Cyclic dependency detected")

    longest_path = _critical_path(graph, indegree.keys())
    return order, longest_path


def _critical_path(graph: Dict[str, List[str]], tasks) -> List[str]:
    memo: Dict[str, Tuple[int, List[str]]] = {}

    def dfs(node: str) -> Tuple[int, List[str]]:
        if node in memo:
            return memo[node]
        if not graph[node]:
            memo[node] = (1, [node])
            return memo[node]
        best_length = 0
        best_path: List[str] = []
        for neighbor in graph[node]:
            length, path = dfs(neighbor)
            if length > best_length:
                best_length = length
                best_path = path
        memo[node] = (best_length + 1, [node] + best_path)
        return memo[node]

    best_overall = (0, [])
    for task in tasks:
        length, path = dfs(task)
        if length > best_overall[0]:
            best_overall = (length, path)
    return best_overall[1]

