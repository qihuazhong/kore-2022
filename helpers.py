import random
from collections import defaultdict
from functools import lru_cache
from typing import List, Dict, Tuple, Union

# <--->
import numpy as np
# from line_profiler_pycharm import profile
from matplotlib import pyplot as plt

from basic import collection_rate
from geometry import Point
from quickboard import QuickBoard, Player, BoardRoute, PlanRoute, Fleet, MiningRoute, SuicideRoute, Shipyard, \
    AttackOpportunity


# <--->

def absorbables(sy: Shipyard, f: Fleet, dest: Point):
    avail_ships_next_step = min(sy.max_ships_to_spawn, sy.player.available_kore() // sy.board.spawn_cost) \
                            + sy.available_ships + sum(sy.incoming_allied_ships[1:2])

    avail_ships_step_2 = min(sy.max_ships_to_spawn*2, (sy.player.available_kore() + sy.player.expected_kore_by_time[1]) // sy.board.spawn_cost) \
                            + sy.available_ships + sum(sy.incoming_allied_ships[1:4])

    avail_ships_step_3 = min(sy.max_ships_to_spawn*3, (sy.player.available_kore() + sum(sy.player.expected_kore_by_time[1:3])) // sy.board.spawn_cost) \
                            + sy.available_ships + sum(sy.incoming_allied_ships[1:4])

    current_step = []
    next_step = []
    step_2 = []
    step_3 = []
    if avail_ships_next_step <= f.ship_count:
        return current_step, next_step, step_2, step_3

    dx, dy = sy.point.deltas_from(dest)

    for time, point in enumerate(f.route.points(), 1):
        if point == sy.point:
            continue

        if sy.available_ships > f.ship_count and f.planned_absorbed_current_step in [sy, None] and sy.point.distance_from(point) == time:
            dx_1, dy_1 = point.deltas_from(sy.point)
            dx_2, dy_2 = point.deltas_from(dest)
            if dx_1 <= dx and dx_2 <= dx and dy_1 <= dy and dy_2 <= dy:
                current_step.append((time, point))
                f.planned_absorbed_current_step = sy
        elif avail_ships_next_step > f.ship_count and f.planned_absorbed_next_step in [sy, None] and sy.point.distance_from(point) == time-1:
            dx_1, dy_1 = point.deltas_from(sy.point)
            dx_2, dy_2 = point.deltas_from(dest)
            if dx_1 <= dx and dx_2 <= dx and dy_1 <= dy and dy_2 <= dy:
                next_step.append((time, point))
                f.planned_absorbed_next_step = sy
        elif avail_ships_step_2 > f.ship_count and f.planned_absorbed_step_2 in [sy, None] and sy.point.distance_from(point) == time-2:
            dx_1, dy_1 = point.deltas_from(sy.point)
            dx_2, dy_2 = point.deltas_from(dest)
            if dx_1 <= dx and dx_2 <= dx and dy_1 <= dy and dy_2 <= dy:
                step_2.append((time, point))
                f.planned_absorbed_step_2 = sy
        elif avail_ships_step_3 > f.ship_count and f.planned_absorbed_step_3 in [sy, None] and sy.point.distance_from(point) == time-3:
            dx_1, dy_1 = point.deltas_from(sy.point)
            dx_2, dy_2 = point.deltas_from(dest)
            if dx_1 <= dx and dx_2 <= dx and dy_1 <= dy and dy_2 <= dy:
                step_3.append((time, point))
                f.planned_absorbed_step_3 = sy

    return current_step, next_step, step_2, step_3


def is_protected_route(board: QuickBoard, route: Union[BoardRoute, MiningRoute],
                       net_power: np.ndarray, num_ships: int, horizon: int = 21
                       ) -> bool:
    if isinstance(route, BoardRoute):
        points = route.points()[:horizon]
    elif isinstance(route, MiningRoute):
        points = board.route_points(route)[:horizon]

    for t, p in enumerate(points, 1):
        if net_power[t][p.x][p.y] + num_ships <= 0:
            return False
    return True


def route_danger_sparse(board: QuickBoard, route: MiningRoute, net_power: np.ndarray, horizon: int = 21) -> bool:

    if horizon <= 0:
        return 0

    time, rows, cols = route.sparse_rep
    return (net_power[time[:horizon] + 1, rows[:horizon], cols[:horizon]]).min()


def get_hostile_fleets(fleet: Fleet, player: Player) -> List[Fleet]:
    hostile_fleets = []

    for time, point in enumerate(fleet.route.points()):

        for opponent_fleet in player.board.fleets:
            if opponent_fleet.player_id == player.game_id:
                continue

            if time < len(opponent_fleet.route.points()):
                opponent_fleet_point = opponent_fleet.route.points()[time]
                if point in opponent_fleet_point.adjacent_points + [opponent_fleet_point]:
                    hostile_fleets.append(opponent_fleet)

    return list(set(hostile_fleets))


def fleet_remaining_health(fleet: Fleet, player: Player
                           ) -> List[Dict[str, Union[Point, int]]]:
    """
    time -> point, health (remaining ship count)
    """
    board = player.board
    allied_shipyard_points = {x.point for x in board.shipyards if x.player_id == player.game_id}

    time_to_fleet_health = []

    remaining_health = fleet.ship_count
    # allied_fleets = [f for f in player.fleets if f != fleet]
    for time, point in enumerate(fleet.route.points()):

        if point in allied_shipyard_points:
            return time_to_fleet_health

        collied_fleets = [af for af in player.fleets if len(af.route.points()) > time]
        collied_fleets = [af for af in collied_fleets if af.route.points()[time] == point]
        if len(collied_fleets) >= 2:
            collied_fleets.sort(key=lambda f: (f.ship_count, f.kore, -f.direction.to_index()), reverse=True)
            if collied_fleets[0] == fleet:
                remaining_health += sum(af.ship_count for af in collied_fleets[1:])
            else:
                # absorbed, just return
                return time_to_fleet_health

        dmg = 0
        # hostile_fleets = []
        for pl in board.players:
            is_enemy = pl != player

            # print(pl.expected_fleets_positions[time][point])
            if is_enemy:
                # adjacent dmg & direct dmg
                for hostile_fleet in pl.fleets:

                    if time < len(hostile_fleet.route.points()):
                        hostile_fleet_point = hostile_fleet.route.points()[time]
                        if point in hostile_fleet_point.adjacent_points + [hostile_fleet_point]:
                            dmg += hostile_fleet.ship_count
                            # hostile_fleets.append(fleet)
                            # print(f'combined dmg {hostile_fleet} {fleet.ship_count} from point {hostile_fleet_point}')

        # assert dmg == dmg_g
        remaining_health -= dmg

        time_to_fleet_health.append({'point': point, 'health': remaining_health})

    return time_to_fleet_health


# @profile
def is_intercept_route(route: Union[BoardRoute, SuicideRoute], player: Player, safety=True,
                       allow_shipyard_intercept=False, allow_friendly_fleet=False, max_sustained_dmg=0) -> bool:
    """ If the route will be intercepted by a fleet(enemy or not) or a shipyard.

    Args:
        route:
        player:
        safety: if `True`, a point receiving damages will also be considered "intercepted"
        allow_shipyard_intercept:

    Returns:

    """
    board = player.board

    if not allow_shipyard_intercept:
        shipyard_points = {x.point for x in board.shipyards}
    else:
        shipyard_points = {}

    if isinstance(route, SuicideRoute):
        points = board.route_points(route)
    elif isinstance(route, BoardRoute):
        points = route.points()

    sustained_dmg = 0
    for time, point in enumerate(points[:-1]):
        if point in shipyard_points:
            return True

        for pl in board.players:
            is_enemy = pl != player

            if is_enemy or (not is_enemy and not allow_friendly_fleet):

                if point in pl.expected_fleets_positions[time]:
                    return True

            # TODO can also check whose fleet will lose more kore when going through dmg position
            if safety and is_enemy:
                if point in pl.expected_dmg_positions[time]:
                    sustained_dmg += pl.expected_dmg_positions[time][point]
                    # return True

    if sustained_dmg > max_sustained_dmg:
        return True

    return False


def sustained_dmg(player: Player, route: SuicideRoute) -> int:
    board = player.board
    points = board.route_points(route)

    sustained_dmg = 0
    for time, point in enumerate(points[:-1]):
        for pl in board.players:
            is_enemy = pl != player

            if is_enemy:
                if point in pl.expected_fleets_positions[time] or point in pl.expected_dmg_positions[time]:
                    sustained_dmg += pl.expected_dmg_positions[time][point]

    return sustained_dmg

# @profile
def find_non_intercept_routes(routes: List[MiningRoute], player: Player, safety: bool = True, exclude=None):
    board = player.board
    # obstacles = board.obstacles_np.copy()
    obstacles = player.obstacles_np.copy()
    max_t = obstacles.shape[0]

    intercept = np.zeros(shape=(len(routes,)), dtype=bool)
    if safety:
        for pl in board.players:
            if pl != player:
                expected_dmg_positions = pl.expected_dmg_positions_np.astype(bool)
                horizon = min(max_t, expected_dmg_positions.shape[0])

                for t in range(horizon):
                    obstacles[t] = obstacles[t] | expected_dmg_positions[t]

    if exclude is not None:
        for time, point in exclude:
            obstacles[time, point.x, point.y] = False

    for idx, route in enumerate(routes):
        time, rows, cols = route.sparse_rep
        max_t = min(obstacles.shape[0], len(time), 10)

        if (obstacles[time[:max_t], rows[:max_t], cols[:max_t]]).any():
            intercept[idx] = True

    return ~intercept


def scale_down_mining_fleet(ori_size: int) -> int:
    if ori_size >= 149:
        scaled = 149
    elif ori_size >= 91:
        scaled = 91
    elif ori_size >= 55:
        scaled = 55
    elif ori_size >= 34:
        scaled = 34
    elif ori_size >= 21:
        scaled = 21
    elif ori_size >= 13:
        scaled = 13
    elif ori_size >= 8:
        scaled = 8
    else:
        scaled = ori_size

    return min(scaled + int(random.randint(0, 2)), ori_size)


def find_shortcut_routes(
        board: QuickBoard,
        start: Point,
        end: Point,
        player: Player,
        num_ships: int,
        safety: bool = True,
        allow_shipyard_intercept=False,
        route_distance=None,
        must_pass=None

) -> List[BoardRoute]:
    # TODO permutations
    if route_distance is None:
        route_distance = start.distance_from(end)
    routes = []
    for p in board:
        distance = start.distance_from(p) + p.distance_from(end)
        if distance != route_distance:
            continue

        path1 = start.dirs_to(p)
        path2 = p.dirs_to(end)
        random.shuffle(path1)
        random.shuffle(path2)

        plan = PlanRoute(path1 + path2)

        if num_ships < plan.min_fleet_size():
            continue

        route = BoardRoute(start, plan)

        if is_intercept_route(
                route,
                player,
                safety=safety,
                allow_shipyard_intercept=allow_shipyard_intercept,
                allow_friendly_fleet=True
        ):
            continue

        if must_pass is not None:
            time, point = must_pass[0]
            if (time, point) not in [(t, p) for t, p in enumerate(route.points(), 1)]:
                continue

        routes.append(route)

    return routes


def is_invitable_victory(player: Player):
    if not player.opponents:
        return True

    board = player.board
    if board.steps_left > 100:
        return False

    board_kore = sum(x.kore for x in board) * (1 + board.regen_rate) ** board.steps_left

    player_kore = player.kore + player.fleet_expected_kore()
    opponent_kore = max(x.kore + x.fleet_expected_kore() for x in player.opponents)
    return player_kore > opponent_kore + board_kore

def get_attacked_risks(agent: Player, f: Fleet):
    attacked_risks = np.zeros(shape=len(f.route)+1, dtype=np.int)
    if not agent.opponents:
        return attacked_risks

    power_pessi = agent.get_net_power_wo_damage
    for t, p in enumerate(f.route.points(), 1):
        if t < min(15, power_pessi.shape[0]):
            attacked_risks[t] = max(0, -power_pessi[t][p.x][p.y])

    # if f.game_id =='72-1':
    #     print(attacked_risks)
    #
    #     for t in range(1, 13):
    #         fig = plt.figure(figsize=(16, 12))
    #         ax = fig.add_subplot(111)
    #         ax.imshow(power_pessi[t].T, cmap='coolwarm')
    #         for (i, j), label in np.ndenumerate(power_pessi[t]):
    #             ax.text(i, j, label, ha='center', va='center')
    #         plt.gca().invert_yaxis()

    # op_shipyards = agent.opponents[0].shipyards
    #
    # for t, p in enumerate(f.route.points(), 1):
    #     for op_sy in op_shipyards:
    #         if op_sy.point.distance_from(p) <= t+1:
    #             # print('xx', t, p)
    #             attack_power = op_sy.estimated_ship_counts((6, True, False, True))
    #             attack_power[0] = op_sy.ship_count
    #             attacked_risks[t] = max(attack_power[: (t + 1) - op_sy.distance_from(p)+1].max(), attacked_risks[t])

    return attacked_risks

def is_better_than_mining(board: QuickBoard, agent: Player,
                          ao: AttackOpportunity,
                          attack_route: BoardRoute,
                          target_fleet: Fleet) -> bool:
    op_loss = target_fleet.route.expected_kore_np(board, target_fleet.ship_count) * collection_rate(target_fleet.ship_count)
    if ao.collision:
        loot_factor = 1.0
    else:
        loot_factor = 0.5

    loot_gain = loot_factor * target_fleet.route.expected_kore_np(board, target_fleet.ship_count, ao.target_time - 1) * collection_rate(target_fleet.ship_count) + target_fleet.kore
    mining_on_the_way = attack_route.expected_kore_np(board, ao.num_ships_to_launch) * collection_rate(ao.num_ships_to_launch)
    # net_loot_gain = loot_gain + op_loss

    mining_gain_per_turn = agent.efficiency_tracker.get_efficiency(ao.departure) * collection_rate(
        ao.num_ships_to_launch)
    expected_mining_gain = mining_gain_per_turn * len(attack_route)

    movable_asset_ratio_attack = (agent.movable_asset + loot_gain + mining_on_the_way) / (
            sum(op.movable_asset for op in agent.opponents) - op_loss)
    movable_asset_ratio_mining = (agent.movable_asset + expected_mining_gain) / sum(
        op.movable_asset for op in agent.opponents)

    if movable_asset_ratio_attack > movable_asset_ratio_mining:
        return True
    else:
        print(
            f'Step {board.step}. {ao.departure}->{target_fleet}: Not worth attacking op_loss:{op_loss:.1f}, '
            f'loot_gain:{loot_gain:.1f},'
            f' mining_on_the_way:{mining_on_the_way:.1f} expected_mining_gain:{expected_mining_gain:.1f}')
        return False


class RouteCache:

    __slots__ = 'departure_x', 'departure_y', 'path', 'return_xy_pairs', 'hash_key', 'out_command', 'return_commands'

    def __init__(self):
        self.departure_x = None
        self.departure_y = None
        # self.xy_pairs = None
        self.path = None
        self.return_xy_pairs = None
        # self.idx = None
        self.hash_key = None
        self.out_command = None
        self.return_commands = None

    # @profile
    def get_route(self,  return_xy_pairs, path, hash_key, return_commands):

        # self.out_command, self.departure_x, self.departure_y, _, _ = hash_key
        # self.departure_x = departure_x
        # self.departure_y = departure_y
        self.path = path
        self.return_xy_pairs = return_xy_pairs
        # self.out_command = out_command
        self.return_commands = return_commands

        # self.departure_x, self.departure_y, self.path, self.return_xy_pairs, self.out_command, self.return_commands = departure_x, departure_y, path, return_xy_pairs, out_command, return_commands

        return self.mining_routes(hash_key)

    @lru_cache(maxsize=1048576 * 2)
    # @profile
    def mining_routes(self, hash_key):
        out_command, departure_x, departure_y, _, _ = hash_key

        out_xy_pairs = ((departure_x, departure_y) + self.path)

        route_pair = [MiningRoute(departure_x=departure_x,
                                  departure_y=departure_y,
                                  xy_pairs=np.concatenate([out_xy_pairs, self.return_xy_pairs[i]]) % 21,
                                  out_command=out_command,
                                  return_command=self.return_commands[i]
                                  )
                      for i in range(len(self.return_xy_pairs))
                      ]
        return route_pair


class MiningEfficiencyTracker:
    def __init__(self):
        self.smoothing_exponent = 0.6
        self.efficiency_history = {}

    def update(self, shipyard: Shipyard, player: Player, route: MiningRoute):

        shipyard_id = shipyard.game_id
        covered_kore = route.expected_kore_sparse(player, 42)
        efficiency = covered_kore / len(route)

        if shipyard_id not in self.efficiency_history:
            self.efficiency_history[shipyard_id] = efficiency
        else:
            self.efficiency_history[shipyard_id] = self.efficiency_history[shipyard_id] * (1 - self.smoothing_exponent) \
                                                   + self.smoothing_exponent * efficiency

    def get_efficiency(self, shipyard: Shipyard):
        shipyard_id = shipyard.game_id
        if shipyard_id in self.efficiency_history:
            return self.efficiency_history[shipyard_id]
        else:
            return 0
