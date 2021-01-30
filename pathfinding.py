#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Largely inspired by https://github.com/hbock-42/Pathfinder-AStar/
from typing import List, Tuple, Set, Dict
import math


__all__ = ['PathFinder']


def _distance(start: Tuple[int, int], target: Tuple[int, int]) -> int:
    """Computes the distance between two points using the Manhattan distance"""
    return math.sqrt(abs(start[0] - target[0])**2 + abs(start[1] - target[1])**2)


def _heuristic(start: Tuple[int, int], target: Tuple[int, int]) -> int:
    """Returns the heuristic of A*, aka the Manhattan distance between the current point and the target"""
    return _distance(start, target)


def _get_node_with_best_f_score(open_set: Set[Tuple[int, int]], f_score: Dict[int, float]) -> Tuple[int, int]:
    """Returns the cell having the lowest score"""
    return min(open_set, key=lambda a: f_score[hash(a)])


def _is_walkable(walkable, cell: Tuple[int, int]) -> bool:
    """
    Checks whether a given cell is walkable or not.
    """
    return cell in walkable


def _is_inside(size: Tuple[int, int], cell: Tuple[int, int]) -> bool:
    """
    Checks that all the indexes are in range and not outside of the grid or outside of timing
    """
    for i in range(2):
        if not (size[i] > cell[i] >= 0):
            return False
    return True


def _check_neighbours(walkable, size, neigh: List) -> List:
    """Uses previous functions to determine if a neighbour is walkable."""
    return [n for n in neigh if _is_inside(size, n) and _is_walkable(walkable, n)]


def _retrace_path(current, came_from, _hash_vector_table) -> List[Tuple[int, int]]:
    """
    Returns the actual path to follow. It starts from the end, and goes back up according to the relative parents.
    """
    cell: Tuple[int, int] = current
    path: List[Tuple[int, int]] = [cell]
    while hash(cell) in came_from.keys():
        parent_hash: str = came_from[hash(cell)]
        cell: Tuple[int, int] = _hash_vector_table[parent_hash]
        path.append(cell)

    return path[::-1]


class PathFinder:
    """
    Object used to compute the path needed to go from a point of
    a given grid to another.
    """

    def __init__(self, walkable: Set[Tuple[int, int]], size: Tuple[int, int]):
        """
        Only the grid is needed as a parameter.
        """
        self.walkable: Set[Tuple[int, int]] = walkable.copy()
        self.size: Tuple[int, int] = size

    def find_path(self, start: Tuple[int, int], target: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Returns the path from the end of the arm to the target.
        If there isn't any path, an empty list is returned.
        """

        if start == target:
            return [start]

        path = self._find_path(start, target)

        return path

    def _find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Actual implementation of the A* algorithm.
        """
        # Initiating the various variables needed
        # Hash table is used to create a correspondence between two hashes.
        # Rather than storing tuples, we store their hash representation.
        _hash_vector_table: Dict[int, Tuple[int, int]] = dict()

        closed_set: Set[Tuple[int, int]] = set()
        open_set: Set[Tuple[int, int]] = {start}

        # Dictionary linking cells to their parents.
        came_from: Dict[int, int] = dict()

        # Dictionaries storing g & f scores of each cell
        g_score: Dict[int, float] = dict()
        f_score: Dict[int, float] = dict()

        g_score[hash(start)] = 0
        f_score[hash(start)] = _heuristic(start, end)

        _hash_vector_table[hash(start)] = start

        while len(open_set) > 0:
            # We always get the cell having the lowest score overall
            current: Tuple[int, int] = _get_node_with_best_f_score(open_set, f_score)

            if current == end:
                # Path found, we retrace our steps
                return _retrace_path(current, came_from, _hash_vector_table)

            open_set.remove(current)
            closed_set.add(current)

            for neighbour in self.get_neighbours(current):
                # If the neighbour is considered as closed, we skip the next part
                if neighbour in closed_set:
                    continue

                if neighbour not in open_set:
                    open_set.add(neighbour)

                # If it has not been explored yet, we had it to the tables.
                if hash(neighbour) not in g_score.keys():
                    g_score[hash(neighbour)] = float('inf')
                    _hash_vector_table[hash(neighbour)] = neighbour

                # If it is more expensive to go to that cell from the current one than before, we continue.
                tentative_g_score: float = g_score[hash(current)] + _distance(current, neighbour)
                if tentative_g_score >= g_score[hash(neighbour)]:
                    continue

                # If it cheaper to go to that cell from the current one, we update both scores.
                came_from[hash(neighbour)] = hash(current)
                g_score[hash(neighbour)] = tentative_g_score
                weight: int = 2

                f_score[hash(neighbour)] = g_score[hash(neighbour)] + weight * _heuristic(neighbour, end)

        return []

    def get_neighbours(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Returns the four neighbours of any given cell
        """
        offsets: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1)]
        neighbours: List[Tuple[int, int]] = [(cell[0] + a, cell[1] + b) for a, b in offsets]
        # print(f"Cell{cell} has neihbours {_check_neighbours(self.walkable, self.size, neighbours)}")
        return _check_neighbours(self.walkable, self.size, neighbours)


if __name__ == "__main__":

    example_set = {(0, 0), (1, 0), (0, 1), (2, 1), (0, 2), (1, 2), (3, 2), (2, 3), (4, 3), (5, 3), (4, 1)}
    set_target: Tuple[int, int] = (5, 3)
    set_start = (0, 0)

    finder: PathFinder = PathFinder(example_set, (6, 4))
    print(finder.find_path(set_start, set_target))
