import math
import time

import numpy as np
import pandas as pd

from collections import namedtuple
from operator import attrgetter
from typing import List, Union, Optional

# import sys
# sys.setrecursionlimit(5000000)

df = pd.read_json('../mining_opt.json')
print(df.tail(50))
print(df.head(50))

# print(df)

Route = namedtuple("Route", ['index', 'kre_per_step', 'min_fleet', 'tour_length'])

fleet_to_best_route = {}
tour_len_to_best_route = {}

for fleet_num in range(2, 200):
    ddf = df[df['min_fleet'] <= fleet_num].sort_values(by='kore_per_step', ascending=False).iloc[0]
    fleet_to_best_route[fleet_num] = {'tour_length': ddf['tour_length'], 'kore_per_step': ddf['kore_per_step']}

for tour_len in range(2, 27, 2):
    ddf = df[df['tour_length'] <= tour_len].sort_values(by='kore_per_step', ascending=False).iloc[0]
    tour_len_to_best_route[tour_len] = {'tour_length': ddf['tour_length'], 'kore_per_step': ddf['kore_per_step'], 'min_fleet': ddf['min_fleet']}

print(fleet_to_best_route)


class Node(object):
    def __init__(self, value, step: int, free_ships=None, expected_kore=None, spawned_ships=0, x: str = ''):
        self.mine_full_fleet = None
        self.mine_half_fleet = None
        self.spawn = None  # not selecting the item
        self.max_spawn = 5  #
        self.step = step

        self.x = x  # decision variables
        if expected_kore is None:
            self.expected_kore = np.zeros(shape=(50,))
        else:
            self.expected_kore = expected_kore
        if free_ships is None:
            self.free_ships = np.zeros(shape=(50,))
        else:
            self.free_ships = free_ships
        self.spawned_ships = spawned_ships
        self.bound = None

        self.value = value


class SearchTree(object):
    def __init__(self, root: Node, fleet_to_best_route):
        self.root = root
        self.initial_ships = root.free_ships[0]
        self.spawn_cost = 10
        self.best_obj = 0
        self.best_solution = None
        self.fleet_to_best_route = fleet_to_best_route  # Should be sorted by a heuristic

        self.evaluation_count = 0
        self.max_evaluation = 20000000

    def create_node_spawn(self, node: Node) -> Optional[Node]:
        next_step = node.step + 1
        # if node.expected_kore[node.step] < self.spawn_cost:
        #     return None

        num_spawn = min(node.expected_kore[node.step] // self.spawn_cost, node.max_spawn)

        expected_kore = node.expected_kore.copy()
        # expected_kore[node.step] -= num_spawn * self.spawn_cost
        expected_kore[next_step] = expected_kore[node.step] + expected_kore[next_step] - num_spawn * self.spawn_cost

        free_ships = node.free_ships.copy()
        free_ships[next_step] = node.free_ships[node.step] + node.free_ships[next_step] + num_spawn

        # print(num_spawn)

        spawned_ships = node.spawned_ships + num_spawn

        node_spawn = Node(step=next_step,
                          x=node.x + f'[s-{int(num_spawn)}]',
                          expected_kore=expected_kore,
                          free_ships=free_ships,
                          spawned_ships=spawned_ships,
                          value=spawned_ships * self.spawn_cost + expected_kore[next_step:].sum()
                          )
        node.spawn = node_spawn

        return node.spawn

    def create_node_mine_full(self, node: Node, fleet_to_best_route, initial_ships) -> Optional[Node]:
        next_step = node.step + 1
        # print(f'step {node.step}')
        # print(f'{node.x}')
        # print(node.free_ships[node.step])
        # print(node.free_ships)
        if node.free_ships[node.step] < 2:
            return None

        if not node.mine_full_fleet:
            # create a mine full node

            route = fleet_to_best_route[int(node.free_ships[node.step])]
            expected_kore = node.expected_kore.copy()
            expected_kore[next_step] = expected_kore[node.step] + expected_kore[next_step]

            launched_ships = node.free_ships[node.step]
            ratio = min(math.log(launched_ships) / 20, 0.99) / min(math.log(initial_ships) / 20, 0.99)

            expected_kore[int(node.step + route['tour_length'])] = route['tour_length'] * route['kore_per_step'] * ratio \
                                                                   + route['tour_length'] * route['kore_per_step'] * ratio

            free_ships = node.free_ships.copy()
            # free_ships[node.step] = free_ships[node.step] - launched_ships
            free_ships[next_step] = free_ships[node.step] - launched_ships + node.free_ships[next_step]
            free_ships[int(node.step + route['tour_length'])] = free_ships[int(node.step + route['tour_length'])] + launched_ships

            node_mine_full = Node(step=next_step,
                                  x=node.x + 'f',
                                  expected_kore=expected_kore,
                                  free_ships=free_ships,
                                  spawned_ships=node.spawned_ships,
                                  value=node.spawned_ships * self.spawn_cost + expected_kore[next_step:].sum()
                                  )
            node.mine_full_fleet = node_mine_full

        return node.mine_full_fleet

    def create_node_mine_half(self, node: Node, fleet_to_best_route, initial_ships) -> Optional[Node]:
        next_step = node.step + 1

        if node.free_ships[node.step] < 4:
            return None

        if not node.mine_half_fleet:
            # create a mine full node

            route = fleet_to_best_route[int(node.free_ships[node.step])]
            expected_kore = node.expected_kore.copy()
            expected_kore[next_step] = expected_kore[node.step] + expected_kore[next_step]

            launched_ships = node.free_ships[node.step] // 2
            ratio = min(math.log(launched_ships) / 20, 0.99) / min(math.log(initial_ships) / 20, 0.99)

            expected_kore[int(node.step + route['tour_length'])] = expected_kore[int(node.step + route['tour_length'])] \
                                                                   + route['tour_length'] * route['kore_per_step'] * ratio

            free_ships = node.free_ships.copy()
            # free_ships[node.step] = free_ships[node.step] - launched_ships
            free_ships[next_step] = free_ships[node.step] - launched_ships + node.free_ships[next_step]
            free_ships[int(node.step + route['tour_length'])] = free_ships[int(node.step + route['tour_length'])] + \
                                                                launched_ships

            node_mine_half = Node(step=next_step,
                                  x=node.x + 'h',
                                  expected_kore=expected_kore,
                                  free_ships=free_ships,
                                  spawned_ships=node.spawned_ships,
                                  value=node.spawned_ships * self.spawn_cost + expected_kore[next_step:].sum()
                                  )
            node.mine_half_fleet = node_mine_half

        return node.mine_half_fleet

    def dfs(self, node: Union[None, Node]):
        if node is None:
            return

        if self.evaluation_count > self.max_evaluation:
            print('max evaluation reached')
            return

        next_step = node.step + 1
        if next_step == 18:
            # bottom of the tree (leaf) has been reached
            # print(node.value)
            # print(node.x)
            if node.value > self.best_obj:
                print(node.value)
                self.best_obj = node.value
                self.best_solution = node.x
                self.best_node = node
        else:

            # Evaluate the best bound of the current node
            node.bound = self.evaluate_bound(node)
            if node.bound > self.best_obj:
                # Branch, continue exploring
                node_spawn = self.create_node_spawn(node)
                self.dfs(node_spawn)

                node_mine_full = self.create_node_mine_full(node,
                                                            fleet_to_best_route=fleet_to_best_route,
                                                            initial_ships=self.initial_ships)
                self.dfs(node_mine_full)

                node_mine_half = self.create_node_mine_half(node,
                                                            fleet_to_best_route=fleet_to_best_route,
                                                            initial_ships=self.initial_ships)
                self.dfs(node_mine_half)

            else:
                # Otherwise, cut this branch
                pass

        del node

    def evaluate_bound(self, node: Node):

        self.evaluation_count += 1
        return 100000000

        value = node.value
        remaining_capacity = node.remaining_capacity

        for item_idx in range(node.step + 1, self.item_count):

            item = self.routes[item_idx]

            if item.weight <= remaining_capacity:
                frac = 1
                value += item.value * frac
                remaining_capacity -= item.weight * frac
            else:
                frac = remaining_capacity / item.weight
                value += item.value * frac
                remaining_capacity -= item.weight * frac
                break

        return value


def solve_it():
    tic = time.time()

    free_ships = np.zeros(shape=(100,))
    free_ships[0] = 50

    expected_kore = np.zeros(shape=(100,))
    expected_kore[0] = 50

    root = Node(value=0, free_ships=free_ships, expected_kore=expected_kore, step=0)
    search_tree = SearchTree(root=root, fleet_to_best_route=fleet_to_best_route, )

    search_tree.dfs(root)

    print(search_tree.best_obj)
    print(search_tree.best_solution)
    print(search_tree.best_node.free_ships)
    print(search_tree.best_node.expected_kore)
    print(search_tree.best_node.spawned_ships)
    print(time.time() - tic)


solve_it()
