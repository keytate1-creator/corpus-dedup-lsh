"""
Disjoint-Set (Union-Find) with path compression and union-by-rank.
"""

from __future__ import annotations

from typing import Dict, List, Set, Generic, TypeVar

T = TypeVar("T")


class DisjointSet(Generic[T]):
    """Tracks a set of elements partitioned into a number of disjoint subsets."""

    def __init__(self) -> None:
        self.parent: Dict[T, T] = {}
        self.rank: Dict[T, int] = {}

    def make_set(self, item: T) -> None:
        """Initializes a singleton set with item."""
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: T) -> T:
        """Finds representative of the set containing item, applying path compression."""
        if item not in self.parent:
            self.make_set(item)
            return item

        path: List[T] = []
        curr = item
        while self.parent[curr] != curr:
            path.append(curr)
            curr = self.parent[curr]

        # Path compression
        for node in path:
            self.parent[node] = curr

        return curr

    def union(self, item_a: T, item_b: T) -> None:
        """Merges the sets containing item_a and item_b using union-by-rank."""
        root_a = self.find(item_a)
        root_b = self.find(item_b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1

    def get_clusters(self) -> Dict[T, Set[T]]:
        """Returns mapping from root representatives to all items in that cluster."""
        clusters: Dict[T, Set[T]] = {}
        for item in self.parent:
            root = self.find(item)
            if root not in clusters:
                clusters[root] = set()
            clusters[root].add(item)
        return clusters
