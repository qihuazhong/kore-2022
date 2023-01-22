import itertools
import re

import numpy as np
from typing import Dict, List, Union, Optional, Generator, Tuple
from collections import defaultdict
from kaggle_environments.envs.kore_fleets.helpers import Configuration, Board

# <--->
# from line_profiler_pycharm import profile

from matplotlib import pyplot as plt

from basic import Obj, collection_rate, max_ships_to_spawn, cached_property, \
    create_spawn_ships_command, create_launch_fleet_command, cached_call, min_ship_count_for_flight_plan_len
from geometry import Field, Action, Point, North, South, Convert, PlanPath, PlanRoute, GAME_ID_TO_ACTION
from logger import logger


# <--->

def highlight_cell(x, y, ax=None, **kwargs):
    rect = plt.Rectangle((x - .5, y - .5), 1, 1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


class _ShipyardAction:
    def to_str(self):
        raise NotImplementedError

    def __repr__(self):
        return self.to_str()


class Spawn(_ShipyardAction):
    def __init__(self, ship_count: int):
        self.ship_count = ship_count

    def to_str(self):
        return create_spawn_ships_command(self.ship_count)


class Launch(_ShipyardAction):
    def __init__(self, ship_count: int, route: "MiningRoute"):
        self.ship_count = ship_count
        self.route = route

    def to_str(self):
        # print(self.route.to_str())
        return create_launch_fleet_command(self.ship_count, self.route.to_str())


class DoNothing(_ShipyardAction):
    def __repr__(self):
        return "Do nothing"

    def to_str(self):
        raise NotImplementedError


class BoardPath:
    max_length = 32

    # @profile
    def __init__(self, start: "Point", plan: PlanPath):
        assert plan.num_steps > 0 or plan.direction == Convert

        self._plan = plan

        field = start.field
        x, y = start.x, start.y
        if np.isfinite(plan.num_steps):
            n = plan.num_steps + 1
        else:
            n = self.max_length
        action = plan.direction

        if plan.direction == Convert:
            self._track = []
            self._start = start
            self._end = start
            self._build_shipyard = True
            return

        if action in (North, South):
            track = field.get_column(x, start=y, size=n * action.dy)
        else:
            track = field.get_row(y, start=x, size=n * action.dx)

        self._track = track[1:]
        self._start = start
        self._end = track[-1]
        self._build_shipyard = False

    def __repr__(self):
        start, end = self.start, self.end
        return f"({start.x}, {start.y}) -> ({end.x}, {end.y})"

    def __len__(self):
        return len(self._track)

    @property
    def plan(self):
        return self._plan

    @property
    def points(self):
        return self._track

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end


class BoardRoute:
    # @profile
    def __init__(self, start: "Point", plan: "PlanRoute", msg: str = ''):
        paths = []
        for p in plan.paths:
            path = BoardPath(start, p)
            start = path.end
            paths.append(path)

        self._plan = plan
        self._paths = paths
        self._start = paths[0].start
        self._end = paths[-1].end
        self.msg = msg

    def __repr__(self):
        points = []
        for p in self._paths:
            points.append(p.start)
        points.append(self.end)
        return " -> ".join([f"({p.x}, {p.y})" for p in points])

    def __iter__(self) -> Generator["Point", None, None]:
        for p in self._paths:
            yield from p.points

    def __len__(self):
        return sum(len(x) for x in self._paths)

    def to_str(self):
        return self.plan.to_str()

    def min_fleet_size(self):
        return self.plan.min_fleet_size()

    def points(self) -> List["Point"]:
        points = []
        for p in self._paths:
            points += p.points
        return points

    @property
    def plan(self) -> PlanRoute:
        return self._plan

    def command(self) -> str:
        return self.plan.to_str()

    @property
    def paths(self) -> List[BoardPath]:
        return self._paths

    @property
    def start(self) -> "Point":
        return self._start

    @property
    def end(self) -> "Point":
        return self._end

    def command_length(self) -> int:
        return len(self.command())

    def last_action(self):
        return self.paths[-1].plan.direction

    def expected_kore_np(self, board: "QuickBoard", ship_count: int, max_t=None):
        """
        Improved: consider kore regeneration and collection
        """
        rate = collection_rate(ship_count)
        if rate <= 0:
            return 0

        if not max_t:
            max_t = min(len(self) + 1, board.size * 2)

        growth_rates = board.growth_rates[:max_t]

        kores = np.zeros(shape=(max_t, board.size, board.size))
        for t, p in enumerate(self.points()[:max_t]):
            kores[t][p.x][p.y] = p.kore
            growth_rates[t][p.x][p.y] = 1 - rate

        kores *= growth_rates

        return sum(kores[t][p.x][p.y] for t, p in enumerate(self.points()[:max_t]))


class MiningRoute:
    """
    a light-weight Point collection object used for mining routes evaluation
    """
    idx = np.arange(42)

    # __slots__ = 'departure_x', 'departure_y', 'xy_pairs', 'out_command', 'return_command', '__trail', '__sparse_rep'

    # @profile
    def __init__(self, departure_x: int, departure_y: int,
                 xy_pairs: np.ndarray, out_command=None, return_command=None):
        self.departure_x = departure_x
        self.departure_y = departure_y
        self.xy_pairs = xy_pairs
        self.out_command = out_command
        self.return_command = return_command
        # self.__trail = None
        # self.__sparse_rep = None

    def __len__(self):
        return len(self.xy_pairs)

    @property
    def end(self):
        return self.xy_pairs[-1]

    @cached_property
    def command(self):

        # if self.connect_commands() != self.to_str():
        #     print(f'Warning', self.to_str(), self.connect_commands(), self.out_command, self.return_command)
        #     assert False
        return self.connect_commands()
        # else:
        # return self.to_str()

    def connect_commands(self):
        for idx, c in enumerate(self.out_command[::-1], 1):
            if c in ['W', 'S', 'N', 'E']:
                last_dit = c
                break

        if last_dit == self.return_command[0]:
            prefix, last_dir_len = self.seperate_last_dir_len(self.out_command)
            first_dir, first_dir_len = self.seperate_first_dir_len(self.return_command)
            if first_dir_len == 0:
                return_command_start_idx = 1
            else:
                return_command_start_idx = 1 + len(str(first_dir_len))
            command = prefix + str(last_dir_len + first_dir_len + 1) + self.return_command[return_command_start_idx:]
        else:
            command = self.out_command + self.return_command

        command, last_dir_len = self.seperate_last_dir_len(command)
        return command

        # if command[-1] not in ['E', 'W', 'N', 'S']:
        #     return command[:-1]
        # else:
        #     return command

    def to_str(self):
        last_x, last_y = self.departure_x, self.departure_y
        command = ''  # TODO
        # for p in self.points():
        for x, y in self.xy_pairs:
            if x - last_x in [1, -20]:
                command = self.process_command(command, 'E')
            elif x - last_x in [-1, 20]:
                command = self.process_command(command, 'W')
            elif y - last_y in [1, -20]:
                command = self.process_command(command, 'N')
            elif y - last_y in [-1, 20]:
                command = self.process_command(command, 'S')
            last_x, last_y = x, y

        new_command, last_dir_len = self.seperate_last_dir_len(command)

        # if command[-1] not in ['E', 'W', 'N', 'S']:
        #     old_command = command[:-1]
        # else:
        #     old_command = command
        #
        # if new_command != old_command:
        #     print(f'Warning', new_command, old_command, self.out_command, self.return_command)
        return new_command

    @cached_property
    def min_fleet_size(self):
        return min_ship_count_for_flight_plan_len(len(self.command))

    def seperate_first_dir_len(self, command):
        x = re.search("(^[A-Z])(\d*).*$", command)
        if x.group(2) == '':
            return x.group(1), 0
        return x.group(1), int(x.group(2))

    def seperate_last_dir_len(self, command):
        x = re.search("(^.*[A-Z])(\d*)$", command)
        if x.group(2) == '':
            return x.group(1), 0
        return x.group(1), int(x.group(2))

    def process_command(self, old_command='', next_dir=''):
        if len(old_command) == 0:
            return next_dir

        if old_command[-1] in ['N', 'E', 'S', 'W']:
            if next_dir == old_command[-1]:
                return old_command + '1'
            else:
                return old_command + next_dir
        else:
            prefix, last_dir_len = self.seperate_last_dir_len(old_command)
            if prefix[-1] == next_dir:
                return prefix + str(last_dir_len + 1)
            else:
                return old_command + next_dir

    # @profile
    def expected_kore_np(self, board: "QuickBoard", ship_count: int):
        """
        Improved: consider kore regeneration and collection
        """
        rate = collection_rate(ship_count)
        if rate <= 0:
            return 0
        # max_t = min(len(self) + 1, board.size * 2)
        #
        # growth_rates = board.growth_rates[:max_t].copy()
        # collected = 0
        # kores = board.kore_arr[:max_t].copy()
        # points = board.route_points(self)
        # for t, p in enumerate(points[:-1]):
        #     collected += kores[t][p.x][p.y] * growth_rates[t][p.x][p.y] * (1 - rate)

        projected_kore = board.projected_kore.copy()
        route_trail = self.trail
        max_t = min(projected_kore.shape[0], route_trail.shape[0] - 1)
        collected = (projected_kore[:max_t] * self.trail[:max_t]).sum()
        return collected
        # return sum(kores[t][p.x][p.y] for t, p in enumerate(self.points()[:-1]))

    # @profile
    def expected_kore_sparse(self, player: "Player", ship_count: int):
        """
        Improved: consider kore regeneration and collection
        """

        idx, rows, cols = self.sparse_rep
        return (player.adjusted_kore_arr[idx[:-1] + 1, rows[:-1], cols[:-1]]).sum()

    @cached_property
    def sparse_rep(self):

        rows, cols = self.xy_pairs.T
        time = self.idx[:len(self)]

        return time, rows, cols


class SuicideRoute(MiningRoute):

    def __init__(self, departure_x: int, departure_y: int, xy_pairs: np.ndarray):
        self.departure_x = departure_x
        self.departure_y = departure_y
        self.xy_pairs = xy_pairs


class PositionObj(Obj):
    def __init__(self, *args, point: Point, player_id: int, board: "QuickBoard", **kwargs):
        super().__init__(*args, **kwargs)
        self._point = point
        self._player_id = player_id
        self._board = board

    # def __repr__(self):
    #     return f"{self.__class__.__name__}(id={self._game_id}, position={self._point}, player={self._player_id})"

    def dirs_to(self, obj: Union["PositionObj", Point]):
        if isinstance(obj, Point):
            return self._point.dirs_to(obj)
        return self._point.dirs_to(obj.point)

    def distance_from(self, obj: Union["PositionObj", Point]) -> int:
        if isinstance(obj, Point):
            return self._point.distance_from(obj)
        return self._point.distance_from(obj.point)

    @property
    def board(self) -> "QuickBoard":
        return self._board

    @property
    def point(self) -> Point:
        return self._point

    @property
    def player_id(self):
        return self._player_id

    @property
    def player(self) -> "Player":
        return self.board.get_player(self.player_id)


class Shipyard(PositionObj):
    # __slots__ = '_ship_count', '_turns_controlled', '_guard_ship_count', 'action'

    def __init__(self, *args, ship_count: int, turns_controlled: int, expected_build_time: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._ship_count = ship_count
        self._turns_controlled = turns_controlled
        self._guard_ship_count = 0
        self.action: Optional[_ShipyardAction] = None
        self.total_reinforcement_needs = 0
        self.immediate_reinforcement_needs = 0
        self.max_route_len = None
        self.max_mining_distance = None
        self.defend_mode = False
        self.reinforcement_mode = False
        self.reinforcement_target = None
        self.action_priority = 0
        self.expected_build_time = expected_build_time
        # self.controlled_points_arr = np.zeros(shape=(self.board.size, self.board.size), dtype=bool)
        self.controlled_points_offence_arr = np.zeros(shape=(self.board.size, self.board.size), dtype=bool)
        # self.power_deficit = None
        self.incoming_hostile_time = None
        self.evacuate = False
        self.is_supply_depot = False
        self.must_pass = None
        self.mining_destination = None
        self.last_attempt_attack = False
        self.solo_capturer = False
        self.recaptured = False

    @property
    def turns_controlled(self):
        return self._turns_controlled

    @property
    def max_ships_to_spawn(self) -> int:
        return max_ships_to_spawn(self._turns_controlled)

    @property
    def ship_count(self):
        return self._ship_count

    @property
    def available_ships(self):
        return self._ship_count - self._guard_ship_count

    @cached_property
    def frontier_risk(self):
        """measure how close the  shipyard is to the battle frontier (opponent's shipyard) """
        player_id = self.player_id
        op_shipyards = [sy for sy in self.board.shipyards if sy.player_id != player_id]
        return sum(1 for sy in op_shipyards if sy.point.distance_from(self.point) <= 10)

    @cached_property
    def frontier_proximity(self):
        player_id = self.player_id
        op_shipyards_dist = [sy.distance_from(self) for sy in self.board.shipyards if sy.player_id != player_id]
        return sum(sorted(op_shipyards_dist)[:2])

    @property
    def guard_ship_count(self):
        return self._guard_ship_count

    def set_guard_ship_count(self, ship_count):
        assert ship_count <= self._ship_count
        self._guard_ship_count = ship_count

    @cached_property
    def incoming_allied_fleets(self) -> List["Fleet"]:
        fleets = []
        for f in self.board.fleets:
            if f.player_id == self.player_id and f.route.end == self.point and self.game_id != f.game_id + 'C':
                fleets.append(f)
        return fleets

    @cached_property
    def incoming_hostile_fleets(self) -> List["Fleet"]:
        fleets = []
        for f in self.board.fleets:
            if f.player_id != self.player_id and f.route.end == self.point:
                fleets.append(f)
        return fleets

    @cached_property
    def incoming_kore(self) -> np.ndarray:
        max_t = self.board.size * 2
        incoming_kore = np.zeros(shape=(max_t,))

        for f in self.incoming_allied_fleets:
            incoming_kore[min(max_t - 1, len(f.route))] += f.expected_kore()
        return incoming_kore

    def likely_wait(self) -> bool:
        spawn_cost = self.player.board.configuration.spawn_cost
        if (
                self.player.kore // spawn_cost >= self.max_ships_to_spawn
                and self.incoming_allied_ships[1] <= 0
        ):
            return True
        elif (
                self.ship_count < 21
                and self.ship_count + self.incoming_allied_ships[1] + min(
            self.max_ships_to_spawn, self.player.kore // spawn_cost) >= 21
        ):
            return True
        else:
            return False

    @cached_call
    def estimated_ship_counts(self, params) -> np.ndarray:
        """
        index 0: current step
        index 1: next step
        """
        # TODO
        patience, offense, is_initiating_shipyard, wait = params

        spawn_cost = self.player.board.configuration.spawn_cost

        potential_spawned = np.array([max_ships_to_spawn(self.turns_controlled + t - 1) for t in range(22)])
        potential_spawned[0] = 0
        potential_spawned[1] = min(self.player.kore // spawn_cost, self.max_ships_to_spawn)

        estimated_ship_counts = np.zeros(shape=(22,))
        estimated_ship_counts[0] = 0  # self.ship_count
        if is_initiating_shipyard:
            estimated_ship_counts[1] = 0
        else:
            if (
                    # (offense and self.likely_wait())
                    # or wait
                    wait
            ):
                estimated_ship_counts[1] = self.ship_count
            else:
                estimated_ship_counts[1] = 0

        for i in range(1, 21):
            estimated_ship_counts[i] += sum(f.final_ship_count for f in self.incoming_allied_fleets
                                            if f.eta == i and not f.attacked)
            estimated_ship_counts[i] -= sum(f.final_ship_count for f in self.incoming_hostile_fleets
                                            if f.eta == i and not f.attacked)

        # estimated_ship_counts += potential_spawned
        max_t = 26
        incoming = np.array([sum(estimated_ship_counts[max(0, i + 1 - patience):i + 1]) for i in range(max_t)])
        spawned = np.array([min(sum(potential_spawned[max(0, i + 1 - patience):i + 1]),
                                (sum(self.player.expected_kore_by_time[:i]) + self.player.kore) // spawn_cost)
                            for i in range(max_t)])

        ship_counts = incoming + spawned
        if self.expected_build_time > 0:
            ship_counts[:self.expected_build_time+1] = 0
        return ship_counts

    @cached_property
    def incoming_allied_ships(self):
        """
        index 0: current step
        index 1: next step
        """
        incoming_allied_ship_counts = np.zeros(shape=(self.board.size + 1,))
        for i in range(1, self.board.size + 1):
            incoming_allied_ship_counts[i] = sum(f.final_ship_count for f in self.incoming_allied_fleets if f.eta == i)

        return incoming_allied_ship_counts

    @cached_call
    # @profile
    def control_range(self, patience_offense_is_initiating_shipyard_wait) -> np.ndarray:
        """
        index 0: current step
        index 1: next step
        """

        patience, offense, is_initiating_shipyard, wait = patience_offense_is_initiating_shipyard_wait
        estimated_ship_counts = self.estimated_ship_counts((patience, offense, is_initiating_shipyard, wait)).copy()

        # print(self, estimated_ship_counts)
        if not offense:
            estimated_ship_counts[0] = 0
            if is_initiating_shipyard:
                estimated_ship_counts[1] = 0  # -= self.available_ships

        control_range = np.zeros(shape=(22, self.board.size, self.board.size))
        control_range[1][self.point.x][self.point.y] = estimated_ship_counts[0]
        control_range[2][self.point.x][self.point.y] = estimated_ship_counts[1]
        for p in self.point.adjacent_points:
            control_range[1][p.x][p.y] = estimated_ship_counts[0]
            control_range[2][p.x][p.y] = estimated_ship_counts[1]

            if offense:
                for ap in p.adjacent_points:
                    control_range[1][ap.x][ap.y] = estimated_ship_counts[0]
                    control_range[2][ap.x][ap.y] = estimated_ship_counts[1]
        # print(offense_initiating_shipyard)
        # print(f'{self} {estimated_ship_counts}')

        if offense:
            offset = 1
            for t in range(3, 22):
                for distance in range(1, t + offset):
                    control_range[t][self.point.x][self.point.y] = estimated_ship_counts[t - 1]

                    perimeter_arr = self.point.perimeter_points_arr(distance)
                    if t - distance >= 0:
                        control_range[t] = np.maximum(control_range[t],
                                                      perimeter_arr * estimated_ship_counts[1:t - distance + 2].max())
                        # control_range[t] += perimeter_arr * estimated_ship_counts[1:t - distance + 2].max()

            # controlled_points_arr = np.repeat(np.expand_dims(self.controlled_points_offence_arr, 0), 22, 0)
            # control_range = control_range * controlled_points_arr
        else:
            offset = -1
            for t in range(3, 22):
                for distance in range(1, t + offset):
                    control_range[t][self.point.x][self.point.y] = estimated_ship_counts[t - 1]

                    perimeter_arr = self.point.perimeter_points_arr(distance)
                    if t - distance >= 0:
                        # control_range[t] = perimeter_arr * estimated_ship_counts[1:t - distance + 2].max()

                        control_range[t] += perimeter_arr * estimated_ship_counts[1:t - distance + 2].max()

                        # if estimated_ship_counts[1: t - distance + 2].min() < 0:
                        #     control_range[t] = np.maximum(control_range[t],
                        #                                   perimeter_arr * estimated_ship_counts[
                        #                                                   1: t - distance + 2].min())
                        # else:
                        #     control_range[t] = np.maximum(control_range[t],
                        #                                   perimeter_arr * estimated_ship_counts[
                        #                                                   1: t - distance + 2].max())

            controlled_points_arr = np.repeat(np.expand_dims(self.controlled_points_offence_arr, 0), 22, 0)
            control_range = control_range * controlled_points_arr

        # if self.game_id == '173-3':
        #     print(control_range[5])

        if offense:
            return control_range
        else:
            return control_range * (abs(control_range) > 4)

    def set_max_route_len(self, max_route_len: int):
        self.max_route_len = max_route_len

    def set_max_mining_distance(self, max_mining_distance):
        self.max_mining_distance = max_mining_distance

    # def set_power_deficit(self, deficit: np.ndarray):
    #     self.power_deficit = deficit

    @cached_property
    def takeover_risk(self):
        rolling_net_power = self.player.rolling_net_power
        net_power_subzero_idx = np.where(rolling_net_power[:, self.point.x, self.point.y] < 0)[0]
        # print(self, rolling_net_power[:, self.point.x, self.point.y])
        if len(net_power_subzero_idx) > 0:
            return net_power_subzero_idx.min()
        else:
            return 21

    @cached_property
    def take_over_time(self):
        power_opti, power_pessi = self.player.net_power((6, None))
        net_power_subzero_idx = np.where(power_pessi[:, self.point.x, self.point.y] < 0)[0]
        if len(net_power_subzero_idx) > 0:
            return net_power_subzero_idx.min()
        else:
            return 21


class Fleet(PositionObj):
    # __slots__ = '_ship_count', '_combined_ship_count', '_survived', 'final_ship_count', '_route', '_kore', 'attacked'

    def __init__(
            self,
            *args,
            ship_count: int,
            kore: int,
            route: BoardRoute,
            direction: Action,
            **kwargs,
    ):

        self.planned_absorbed = False
        assert ship_count > 0
        assert kore >= 0

        super().__init__(*args, **kwargs)

        self._ship_count = ship_count
        self._combined_ship_count = self._ship_count
        self._survived = True
        self.final_ship_count = ship_count
        self._kore = kore
        self._direction = direction
        self._route = route
        self.attacked = False
        self.latent_target = False
        self.suicide_attacker = False
        self.absorbed = False
        self.planned_absorbed_current_step = None
        self.planned_absorbed_next_step = None
        self.planned_absorbed_step_2 = None
        self.planned_absorbed_step_3 = None
        self.route_health = []
        self.damage_to = []

    def __gt__(self, other):
        if self.combined_ship_count != other.combined_ship_count:
            return self.combined_ship_count > other.combined_ship_count
        if self.kore != other.kore:
            return self.kore > other.kore
        return self.direction.game_id > other.direction.game_id

    def __lt__(self, other):
        return other.__gt__(self)

    def absorb(self, other: "Fleet"):
        self._combined_ship_count += other.combined_ship_count
        self.final_ship_count += other.final_ship_count
        other.set_not_survived()
        other.set_absorbed()

    def set_not_survived(self, damage=None):
        self._survived = False
        if damage:
            self.final_ship_count -= damage
        else:
            self.final_ship_count = 0

    @property
    def combined_ship_count(self):
        return self._combined_ship_count

    @property
    def ship_count(self):
        return self._ship_count

    @property
    def kore(self):
        return self._kore

    @property
    def route(self) -> BoardRoute:
        return self._route

    @property
    def eta(self):
        return len(self._route)

    def set_route(self, route: BoardRoute):
        self._route = route

    @property
    def direction(self):
        return self._direction

    @property
    def collection_rate(self) -> float:
        return collection_rate(self._ship_count)

    def expected_kore(self):
        return self._kore + self._route.expected_kore_np(self._board, self._ship_count) * collection_rate(
            self._ship_count)

    def cost(self):
        return self.board.spawn_cost * self.ship_count

    def value(self):
        return self.kore / self.cost()

    # @cached_property
    # def loot_value(self):
    #     return self.expected_kore() / self.cost()

    @cached_property
    def loot_value(self):
        return self.expected_kore() + self.suicide_attacker * self.ship_count * self.board.spawn_cost

    def add_route_health(self, point: Point, health: int):
        if self.route_health and self.route_health[-1]['health'] <= 0:
            return

        self.route_health.append({'point': point, 'health': health})

    def set_absorbed(self):
        self.absorbed = True

    @cached_property
    def expected_dmg_positions_np(self) -> np.ndarray:
        """
        time -> point -> dmg
        """
        if self.player.fleets:
            max_t = max(len(f.route) for f in self.player.fleets) + 1
        else:
            max_t = self.board.size * 2

        shipyard_points = [sy.point for sy in self.board.shipyards]

        time_to_dmg_positions = np.zeros(shape=(max_t, self.board.size, self.board.size))

        for time, point_health_dict_ in enumerate(self.route_health):
            point = point_health_dict_['point']

            if point is not None and point not in shipyard_points:
                health = point_health_dict_['health']
                for adjacent_point in point.adjacent_points + [point]:
                    # time_to_dmg_positions[time][adjacent_point.x][adjacent_point.y] += f.ship_count
                    time_to_dmg_positions[time][adjacent_point.x][adjacent_point.y] += max(0, health)

        return time_to_dmg_positions


class AttackOpportunity:
    __slots__ = 'departure', 'target_time', 'target_point', 'num_ships_to_launch', 'collision', 'patience', \
                'must_pass', 'absorbed_ships', 'partial', 'rescue'

    def __init__(self, departure: Shipyard, target_time: int, target_point: Point, num_ships_to_launch: int,
                 collision: bool, patience=1, must_pass=None, absorbed_ships=0, partial=False, rescue=False):
        self.departure = departure
        self.target_time = target_time
        self.target_point = target_point
        self.num_ships_to_launch = num_ships_to_launch
        self.collision = collision
        self.patience = patience
        self.must_pass = must_pass
        self.absorbed_ships = absorbed_ships
        self.partial = partial
        self.rescue = rescue


class FleetPointer:
    def __init__(self, fleet: Fleet):
        self.obj = fleet
        self.point = fleet.point
        self.is_active = True
        self._paths = []
        self._points = self.points()

    def points(self):
        for path in self.obj.route.paths:
            self._paths.append([path.plan.direction, 0])
            for point in path.points:
                self._paths[-1][1] += 1
                yield point

    def update(self):
        if not self.is_active:
            self.point = None
            return
        try:
            self.point = next(self._points)
        except StopIteration:
            self.point = None
            self.is_active = False

    def current_route(self):
        plan = PlanRoute([PlanPath(d, n) for d, n in self._paths])
        return BoardRoute(self.obj.point, plan)


class Player(Obj):
    def __init__(self, *args, kore: float, board: "QuickBoard", **kwargs):
        super().__init__(*args, **kwargs)

        self._kore = kore
        self._board = board
        self.preloaded_mining_routes = None
        self.preloaded_return_routes = None
        self.route_cache = None
        self.efficiency_tracker: "MiningEfficiencyTracker" = None
        self.player_controlled_points_arr = np.zeros(shape=(board.size, board.size), dtype=bool)
        self.player_controlled_points_offence_arr = np.zeros(shape=(board.size, board.size), dtype=bool)

    @property
    def kore(self):
        return self._kore

    def fleet_kore(self):
        return sum(x.kore for x in self.fleets)

    def fleet_expected_kore(self):
        return sum(x.expected_kore() for x in self.fleets)

    @cached_property
    def expected_kore_by_time(self):
        max_t = 26
        expected_kore_by_time = np.zeros(shape=max_t)
        for f in self.fleets:
            if f.eta < 26:
                expected_kore_by_time[f.eta] += f.expected_kore()
        return expected_kore_by_time

    # sum(x.expected_kore() for x in sy.incoming_allied_fleets if x.eta <= 20)

    def is_active(self):
        return len(self.fleets) > 0 or len(self.shipyards) > 0

    @property
    def board(self):
        return self._board

    def _get_objects(self, name):
        d = []
        for x in self._board.__getattribute__(name):
            if x.player_id == self.game_id:
                d.append(x)
        return d

    @cached_property
    def fleets(self) -> List[Fleet]:
        return self._get_objects("fleets")

    @cached_property
    def shipyards(self) -> List[Shipyard]:
        return self._get_objects("shipyards")

    @cached_property
    def ship_count(self) -> int:
        conversion_en_route = self.board.configuration.convert_cost * len(
            [sy for sy in self.shipyards if 'C' in sy.game_id])
        return sum(x.ship_count for x in itertools.chain(self.fleets, self.shipyards)) - conversion_en_route

    @cached_property
    def movable_asset(self):
        return self.ship_count * self.board.spawn_cost + self.kore

    @cached_property
    def opponents(self) -> List["Player"]:
        return [x for x in self.board.players if x != self]

    @cached_property
    def expected_fleets_positions(self) -> Dict[int, Dict[Point, Fleet]]:
        """
        time -> point -> fleet
        """
        time_to_fleet_positions = defaultdict(dict)
        for f in self.fleets:
            for time, point in enumerate(f.route):
                time_to_fleet_positions[time][point] = f
        return time_to_fleet_positions

    @cached_property
    def expected_dmg_positions(self) -> Dict[int, Dict[Point, int]]:
        """
        time -> point -> dmg
        """

        shipyard_points = [sy.point for sy in self.board.shipyards]

        time_to_dmg_positions = defaultdict(dict)
        for f in self.fleets:
            for time, point in enumerate(f.route):
                if point not in shipyard_points:
                    for adjacent_point in point.adjacent_points:
                        point_to_dmg = time_to_dmg_positions[time]
                        if adjacent_point not in point_to_dmg:
                            point_to_dmg[adjacent_point] = 0
                        point_to_dmg[adjacent_point] += f.ship_count
        return time_to_dmg_positions

    @cached_property
    def expected_dmg_positions_np(self) -> np.ndarray:
        """
        time -> point -> dmg
        """
        if self.fleets:
            max_t = max(len(f.route) for f in self.fleets) + 1
        else:
            max_t = self.board.size * 2

        shipyard_points = [sy.point for sy in self.board.shipyards]

        time_to_dmg_positions = np.zeros(shape=(max_t, self.board.size, self.board.size))
        for f in self.fleets:
            # for time, point in enumerate(f.route):
            # print(f.route_health) # TODO optimize
            for time, point_health_dict_ in enumerate(f.route_health):
                point = point_health_dict_['point']

                if point is not None and point not in shipyard_points:
                    health = point_health_dict_['health']
                    for adjacent_point in point.adjacent_points + [point]:
                        # time_to_dmg_positions[time][adjacent_point.x][adjacent_point.y] += f.ship_count
                        if adjacent_point not in shipyard_points:
                            time_to_dmg_positions[time][adjacent_point.x][adjacent_point.y] += max(0, health)

        return time_to_dmg_positions

    # @cached_call
    # def expected_dmg_positions_shifted(self, shift: Tuple) -> np.ndarray:
    #     return np.roll(self.expected_dmg_positions_np, shift=shift, axis=(1, 2))

    def actions(self, test=False):
        if self.available_kore() < 0:
            logger.warning("Negative balance. Some ships will not spawn.")

        shipyard_id_to_action = {}
        for sy in self.shipyards:
            if not sy.action or isinstance(sy.action, DoNothing):
                continue
            if test and isinstance(sy.action, Launch):
                route = shipyard_id_to_action[sy.game_id] = sy.action.route
                if isinstance(route, MiningRoute):
                    shipyard_id_to_action[sy.game_id] = route.end[0], route.end[1]
                else:
                    shipyard_id_to_action[sy.game_id] = route.end.x, route.end.y
            else:
                shipyard_id_to_action[sy.game_id] = sy.action.to_str()
        return shipyard_id_to_action

    def spawn_ship_count(self):
        return sum(
            x.action.ship_count for x in self.shipyards if isinstance(x.action, Spawn)
        )

    def need_kore_for_spawn(self):
        return self.board.spawn_cost * self.spawn_ship_count()

    def available_kore(self):
        return self._kore - self.need_kore_for_spawn()

    @cached_property
    def adjusted_kore_arr(self):
        op = None
        for pl in self.board.players:
            if pl != self:
                op = pl

        adjust_factor = 0.2 + 0.8 * min(1, (self.kore + sum(self.expected_kore_by_time)) / 3000)
        max_t = self.board.projected_kore.shape[0]
        control_adjustments = np.ones_like(op.player_controlled_points_arr, dtype=np.float)
        control_adjustments += op.player_controlled_points_arr * adjust_factor * 0.7
        control_adjustments += op.player_controlled_points_offence_arr * adjust_factor * 0.7
        adjusted_kore_arr = self.board.projected_kore * np.repeat(np.expand_dims(control_adjustments, 0), max_t, 0)

        return adjusted_kore_arr

    # @profile
    def update_controlled_points(self):
        shipyards = self.shipyards

        for p in self.board:
            min_distance = self.point_dist_oppo(p)

            for sy in shipyards:
                # if sy.distance_from(p) <= min_distance:
                #     sy.controlled_points_arr[p.x][p.y] = True
                if sy.distance_from(p) <= min_distance + 3:
                    self.player_controlled_points_arr[p.x][p.y] = True
                if sy.distance_from(p) <= min_distance + 2:
                    self.player_controlled_points_offence_arr[p.x][p.y] = True
                if sy.distance_from(p) <= min_distance + 1:
                    sy.controlled_points_offence_arr[p.x][p.y] = True

    @cached_call
    # @profile
    def get_control_map(self, patience_offense_initiating_shipyard):
        """
        index 0: current step
        index 1: next step
        """
        patience, offense, initiating_shipyard = patience_offense_initiating_shipyard

        friendly_power_min = np.zeros(shape=(22, self.board.size, self.board.size))
        friendly_power_max = np.zeros(shape=(22, self.board.size, self.board.size))
        for sy in self.shipyards:
            if initiating_shipyard == sy:
                is_initiating_shipyard = True
            else:
                is_initiating_shipyard = False

            sy_control_range_min = sy.control_range((patience, offense, is_initiating_shipyard, False))
            sy_control_range_max = sy.control_range((patience, offense, is_initiating_shipyard, True))
            friendly_power_min += sy_control_range_min
            friendly_power_max += sy_control_range_max

        if offense:
            pad_right = max(0, 22 - 1 - self.expected_dmg_positions_np.shape[0])
            friendly_power_min += np.concatenate([np.zeros(shape=(1, self.board.size, self.board.size)),
                                                  self.expected_dmg_positions_np,
                                                  np.zeros(shape=(pad_right, self.board.size, self.board.size))])[:22]
            friendly_power_max += np.concatenate([np.zeros(shape=(1, self.board.size, self.board.size)),
                                                  self.expected_dmg_positions_np,
                                                  np.zeros(shape=(pad_right, self.board.size, self.board.size))])[:22]
        return friendly_power_min, friendly_power_max

    @cached_property
    def get_net_power_wo_damage(self):
        _, net_power_wo_damage = self.net_power((1, None))

        if not self.opponents:
            return net_power_wo_damage

        op = self.opponents[0]
        pad_right = max(0, 22 - 1 - op.expected_dmg_positions_np.shape[0])
        compensate = np.concatenate([np.zeros(shape=(1, self.board.size, self.board.size)),
                                     op.expected_dmg_positions_np,
                                     np.zeros(shape=(pad_right, self.board.size, self.board.size))])[:22]
        return net_power_wo_damage + compensate

    @cached_call
    # @profile
    def net_power(self, patience_initiating_shipyard: Shipyard = (6, None)) -> np.ndarray:
        """
        """
        debug = False
        patience, initiating_shipyard = patience_initiating_shipyard
        friendly_power_min, friendly_power_max = self.get_control_map((patience, False, initiating_shipyard))

        if not self.opponents:
            return friendly_power_min, friendly_power_min

        for op in self.opponents:
            hostile_power_min, hostile_power_max = op.get_control_map((patience, True, initiating_shipyard))

        if patience < 6:
            power_opti = friendly_power_min - hostile_power_min
            power_pessi = friendly_power_max - hostile_power_max
        else:
            power_opti = friendly_power_min - hostile_power_min
            power_pessi = friendly_power_min - hostile_power_max

        if debug and initiating_shipyard and initiating_shipyard and initiating_shipyard.game_id == '0-1':
            for t in range(1, 15):

                fig = plt.figure(figsize=(16, 12))
                ax = fig.add_subplot(111)
                ax.imshow(power_pessi[t].T, cmap='coolwarm')

                highlight_cell(5, 8, color="limegreen", linewidth=3)

                for (i, j), label in np.ndenumerate(power_pessi[t]):
                    ax.text(i, j, label, ha='center', va='center')
                plt.gca().invert_yaxis()

        return power_opti, power_pessi

    # @profile
    def set_supply_depots(self):
        if not self.shipyards:
            return

        min_takeover_risks = min(sy.takeover_risk for sy in self.shipyards)
        for sy in self.shipyards:
            if sy.takeover_risk >= min_takeover_risks + 5:
                sy.is_supply_depot = True

    @cached_property
    def rolling_net_power(self):
        power_opti, power_pessi = self.net_power((6, None))
        rolling_net_power = np.array([power_pessi[i:i + 3].sum(axis=0) for i in range(len(power_pessi))])

        # for t in range(1, 6):
        #     fig = plt.figure(figsize=(16, 12))
        #     ax = fig.add_subplot(111)
        #     ax.imshow(power_pessi[t].T, cmap='coolwarm')
        #     for (i, j), label in np.ndenumerate(power_pessi[t]):
        #         ax.text(i, j, label, ha='center', va='center')
        #     plt.gca().invert_yaxis()

        return rolling_net_power

    @cached_call
    def point_dist_oppo(self, p):
        oppo_shipyards = [sy for sy in self.board.shipyards if sy.player_id != self.game_id]
        if oppo_shipyards:
            min_distance = min(sy.distance_from(p) - 1 for sy in oppo_shipyards)
        else:
            min_distance = 21

        return min_distance

    @cached_property
    def obstacles_np(self) -> np.ndarray:
        """time dimension -> index 0 -> next step
        """
        obstacles = np.zeros(shape=(self.board.size, self.board.size), dtype=bool)
        for pl in self.board.players:
            if pl == self:
                continue
            for sy in pl.shipyards:
                obstacles[sy.point.x][sy.point.y] = True

        fleet_positions = self.board.fleet_positions_np.astype(bool)
        max_t = fleet_positions.shape[0]
        # fleet_positions = fleet_positions[: max_t]

        obstacles = np.repeat(np.expand_dims(obstacles, 0), max_t, 0)
        obstacles = obstacles | fleet_positions

        return obstacles


_FIELD = None


class QuickBoard:

    # @profile
    def __init__(self, obs, conf):

        self.configuration = Configuration(conf)
        self._step = obs["step"]

        self.discount_rate = min(1. + (self._step - 150) / 150 * 0.005, 1)

        global _FIELD
        if _FIELD is None or self._step == 0:
            _FIELD = Field(self.configuration.size)
        else:
            assert _FIELD.size == self.configuration.size

        self._field: Field = _FIELD

        id_to_point = {x.game_id: x for x in self._field}

        for point_id, kore in enumerate(obs["kore"]):
            point = id_to_point[point_id]
            point.set_kore(kore)

        self._players = []
        self._fleets = []
        self._shipyards = []
        for player_id, player_data in enumerate(obs["players"]):
            player_kore, player_shipyards, player_fleets = player_data
            player = Player(game_id=player_id, kore=player_kore, board=self)
            self._players.append(player)

            for fleet_id, fleet_data in player_fleets.items():
                point_id, kore, ship_count, direction, flight_plan = fleet_data
                position = id_to_point[point_id]
                direction = GAME_ID_TO_ACTION[direction]
                if ship_count < self.shipyard_cost and Convert.command in flight_plan:
                    # can't convert
                    flight_plan = "".join(
                        [x for x in flight_plan if x != Convert.command]
                    )
                plan = PlanRoute.from_str(flight_plan, direction)
                route = BoardRoute(position, plan)
                fleet = Fleet(
                    game_id=fleet_id,
                    point=position,
                    player_id=player_id,
                    ship_count=ship_count,
                    kore=kore,
                    route=route,
                    direction=direction,
                    board=self,
                )
                self._fleets.append(fleet)

                if Convert.command in flight_plan:
                    # Create a placeholder for future shipyard
                    if fleet.route.points():
                        point = fleet.route.points()[-1]
                    else:
                        point = fleet.point
                    shipyard = Shipyard(
                        game_id=fleet_id + 'C',
                        point=point,
                        player_id=player_id,
                        ship_count=fleet.ship_count - self.shipyard_cost,
                        turns_controlled=0,
                        board=self,
                        expected_build_time=len(fleet.route),
                    )
                    shipyard.action = DoNothing()
                    self._shipyards.append(shipyard)

            for shipyard_id, shipyard_data in player_shipyards.items():
                point_id, ship_count, turns_controlled = shipyard_data
                position = id_to_point[point_id]
                shipyard = Shipyard(
                    game_id=shipyard_id,
                    point=position,
                    player_id=player_id,
                    ship_count=ship_count,
                    turns_controlled=turns_controlled,
                    board=self,
                )
                self._shipyards.append(shipyard)

        self._players = [x for x in self._players if x.is_active()]

        self._update_fleets_destination()
        for player in self.players:
            player.update_controlled_points()

    def __getitem__(self, item):
        return self._field[item]

    def __iter__(self):
        return self._field.__iter__()

    @property
    def field(self):
        return self._field

    @property
    def size(self):
        return self._field.size

    @property
    def step(self):
        return self._step

    @property
    def steps_left(self):
        return self.configuration.episode_steps - self._step - 1

    @property
    def shipyard_cost(self):
        return self.configuration.convert_cost

    @property
    def spawn_cost(self):
        return self.configuration.spawn_cost

    @property
    def regen_rate(self):
        return self.configuration.regen_rate

    @property
    def max_cell_kore(self):
        return self.configuration.max_cell_kore

    @property
    def players(self) -> List[Player]:
        return self._players

    @property
    def fleets(self) -> List[Fleet]:
        return self._fleets

    @property
    def shipyards(self) -> List[Shipyard]:
        return self._shipyards

    def get_player(self, game_id) -> Player:
        for p in self._players:
            if p.game_id == game_id:
                return p
        raise KeyError(f"Player `{game_id}` doas not exists.")

    def get_obj_at_point(self, point: Point) -> Optional[Union[Fleet, Shipyard]]:
        for x in itertools.chain(self.fleets, self.shipyards):
            if x.point == point:
                return x

    @cached_call
    def accum_discount_rate(self, player_route_len) -> float:
        player, route_len = player_route_len
        discount_rate = 1
        fleets = player.fleets
        max_spawns = sum(sy.max_ships_to_spawn for sy in player.shipyards)
        spawn_cost = player.board.configuration.spawn_cost
        # expected_kore = np.zeros(shape=(route_len,))
        for i in range(route_len):
            expected_kore = sum(f.expected_kore() for f in fleets if f.eta == i)
            if expected_kore < max_spawns * spawn_cost:
                discount_rate *= self.discount_rate

        return discount_rate

    @cached_property
    def growth_rates(self) -> np.ndarray:
        max_t = self.size * 2
        rates = np.ones(shape=(max_t, self.size, self.size)) * (1 + self.configuration.regen_rate)
        for f in self.fleets:
            for t, p in enumerate(f.route):
                if t < max_t:
                    rates[t][p.x][p.y] *= 1 - f.collection_rate

        for i in range(1, max_t):
            rates[i] *= rates[i - 1]

        return rates

    @cached_property
    def projected_kore(self):
        max_t = min(self.growth_rates.shape[0], self.kore_arr.shape[0])
        return self.growth_rates[:max_t] * self.kore_arr[:max_t]

    @cached_property
    def fleet_positions_np(self) -> np.ndarray:
        """time dimension: 0 -> next step
        """
        if self.fleets:
            max_t = max(len(f.route) for f in self.fleets) + 1
        else:
            max_t = self.size * 2
        positions = np.zeros(shape=(max_t, self.size, self.size))
        for f in self.fleets:
            for t, p in enumerate(f.route):
                if t < max_t:
                    positions[t][p.x][p.y] = f.ship_count

        return positions

    @cached_property
    def obstacles_np(self) -> np.ndarray:
        """time dimension -> index 0 -> next step
        """
        obstacles = np.zeros(shape=(self.size, self.size), dtype=bool)
        for sy in self.shipyards:
            obstacles[sy.point.x][sy.point.y] = True

        fleet_positions = self.fleet_positions_np.astype(bool)
        max_t = fleet_positions.shape[0]
        # fleet_positions = fleet_positions[: max_t]

        obstacles = np.repeat(np.expand_dims(obstacles, 0), max_t, 0)
        obstacles = obstacles | fleet_positions

        return obstacles

    @cached_property
    def low_kore_points(self):
        # for p in self:
        #     if p.kore + sum(adj_p.kore for adj_p in p.adjacent_points) < 20:
        return [p for p in self if p.kore + sum(adj_p.kore for adj_p in p.adjacent_points) < 20]

    # @cached_call
    # def obstacles_shifted(self, shift: Tuple) -> np.ndarray:
    #     return np.roll(self.obstacles_np, shift=shift, axis=(1, 2))

    def _update_fleets_destination(self):
        """
        trying to predict future positions
        very inaccurate
        """

        shipyard_positions = {x.point for x in self.shipyards}

        fleets = [FleetPointer(f) for f in self.fleets]

        while any(x.is_active for x in fleets):
            for f in fleets:
                f.update()

            # fleet to shipyard
            for f in fleets:
                if f.point in shipyard_positions:
                    f.is_active = False

            # allied fleets
            for player in self.players:
                point_to_fleets = defaultdict(list)
                for f in fleets:
                    if f.is_active and f.obj.player_id == player.game_id:
                        point_to_fleets[f.point].append(f)
                for point_fleets in point_to_fleets.values():
                    if len(point_fleets) > 1:
                        sorted_fleets = sorted(point_fleets, key=lambda x: x.obj)
                        for f in sorted_fleets[:-1]:
                            f.is_active = False
                            sorted_fleets[-1].obj.absorb(f.obj)

            # fleets collision
            point_to_fleets = defaultdict(list)
            for f in fleets:
                if f.is_active:
                    point_to_fleets[f.point].append(f)
            for point_fleets in point_to_fleets.values():
                if len(point_fleets) > 1:
                    sorted_fleet = sorted(point_fleets, key=lambda x: x.obj)
                    winning_fleet = sorted_fleet[-1]
                    dmg_to_winning_fleet = sum(f.obj.final_ship_count for f in sorted_fleet[:-1])
                    for f in sorted_fleet[:-1]:
                        f.is_active = False
                        f.obj.set_not_survived(damage=winning_fleet.obj.final_ship_count)
                        # print('yyy', [x.obj for x in point_fleets if x.obj.player_id != f.player_id])
                        f.obj.damage_to += [x.obj for x in point_fleets if x.obj.player_id != f.obj.player_id]

                    winning_fleet.obj.final_ship_count -= dmg_to_winning_fleet

            # adjacent damage
            point_to_fleet = {}
            for f in fleets:
                if f.is_active:
                    point_to_fleet[f.point] = f

            point_to_dmg = defaultdict(int)
            for point, fleet in point_to_fleet.items():
                attacked_count = 0
                for p in point.adjacent_points:
                    if p in point_to_fleet:
                        adjacent_fleet = point_to_fleet[p]
                        if adjacent_fleet.obj.player_id != fleet.obj.player_id:
                            attacked_count += 1
                            point_to_dmg[p] += fleet.obj.final_ship_count
                if attacked_count >= 2:
                    fleet.obj.suicide_attacker = True
                    # print(f'Identified suicide attacker {fleet.point}')

            for point, fleet in point_to_fleet.items():
                dmg = point_to_dmg[point]
                if fleet.obj.ship_count <= dmg:
                    fleet.is_active = False
                    fleet.obj.set_not_survived(dmg)
                    # print('xxxx', [x.obj for x in fleets if x.point in point.adjacent_points])
                    fleet.obj.damage_to += [x.obj for x in fleets if x.point in point.adjacent_points]
                else:
                    fleet.obj.final_ship_count -= dmg

            for f in fleets:
                f.obj.add_route_health(f.point, f.obj.final_ship_count)

        for f in fleets:
            f.obj.set_route(f.current_route())

    @cached_property
    def kore_arr(self) -> np.ndarray:

        kore_arr = np.zeros(shape=(self.size, self.size))
        for p in self:
            kore_arr[p.x][p.y] = p.kore

        return np.repeat(np.expand_dims(kore_arr, 0), 27, 0)

    @cached_call
    def route_points(self, route: MiningRoute):
        return [self.field[x, y] for x, y in route.xy_pairs]
