from math import ceil
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
# from line_profiler_pycharm import profile

# <--->
from matplotlib import pyplot as plt

from basic import min_ship_count_for_flight_plan_len
from geometry import Point, Convert, PlanRoute, PlanPath
from quickboard import Player, BoardRoute, Launch, Shipyard, DoNothing, Spawn


# <--->


# @profile
def expand(player: Player):
    board = player.board
    num_shipyards_to_create = need_more_shipyards(player)
    if not num_shipyards_to_create:
        return

    shipyard_positions = {x.point for x in board.shipyards}

    # shipyard_to_point = find_best_position_for_shipyards(player)
    full_scores = find_best_position_for_shipyards(player)

    shipyard_count = 0

    for shipyard in player.shipyards:
        # if shipyard.action:
        #     continue
        targets = [t for t in full_scores if t['point'].distance_from(shipyard.point) <= 8][:5]
        for target in targets:

            if shipyard.action:
                break

            if shipyard_count >= num_shipyards_to_create:
                break

            if shipyard.takeover_risk <= 7:
                continue

            if shipyard.available_ships < board.shipyard_cost:
                max_spawn = min(player.available_kore() // board.spawn_cost, shipyard.max_ships_to_spawn*3)

                if (
                    shipyard.available_ships + sum(shipyard.incoming_allied_ships[1:4]) + max_spawn >= board.shipyard_cost
                    and shipyard.incoming_allied_ships[1] < board.shipyard_cost
                ):
                    num_to_spawn = min(player.available_kore() // board.spawn_cost, shipyard.max_ships_to_spawn)
                    if num_to_spawn > 0:
                        shipyard.action = Spawn(int(num_to_spawn))
                    else:
                        shipyard.action = DoNothing()
                    continue

            if (
                shipyard.available_ships < board.shipyard_cost
                or shipyard.action
                or shipyard.defend_mode
                or shipyard.reinforcement_mode
            ):
                continue

            incoming_hostile_fleets = shipyard.incoming_hostile_fleets
            if incoming_hostile_fleets:
                continue

            target_distance = shipyard.distance_from(target['point'])

            # power_opti, power_pessi = player.net_power((6, shipyard))

            routes = []
            for p in board:
                if p in shipyard_positions:
                    continue

                distance = shipyard.distance_from(p) + p.distance_from(target['point'])
                if distance > target_distance:
                    continue

                plan = PlanRoute(shipyard.dirs_to(p) + p.dirs_to(target['point']))
                route = BoardRoute(shipyard.point, plan)

                if shipyard.available_ships < min_ship_count_for_flight_plan_len(
                        len(route.plan.to_str()) + 1
                ):
                    continue

                route_points = route.points()
                # if any(x in shipyard_positions for x in route_points):
                #     continue

                if not is_safety_route_to_convert(route_points, player):
                    continue

                routes.append(route)

            if routes:
                routes.sort(key=lambda r: r.expected_kore_np(board, shipyard.available_ships), reverse=True)
                # route = random.choice(routes)
                route = routes[0]

                route = BoardRoute(
                    shipyard.point, route.plan + PlanRoute([PlanPath(Convert)])
                )
                shipyard.action = Launch(shipyard.available_ships, route)
                print(f'Step {board.step} Launch for new shipyard at {target["point"]}')

                shipyard = Shipyard(
                    game_id=shipyard.game_id + 'C',
                    point=route.points()[-1],
                    player_id=player.game_id,
                    ship_count=shipyard.available_ships - board.shipyard_cost,
                    turns_controlled=0,
                    board=board,
                    expected_build_time=len(route),
                )
                shipyard.action = DoNothing()
                board._shipyards.append(shipyard)

                shipyard_count += 1


# @profile
def find_best_position_for_shipyards(player: Player) -> List[Tuple[Shipyard, Point]]:
    board = player.board
    shipyards = board.shipyards

    area_per_shipyard = board.size ** 2 / (len(board.shipyards) + 1)
    radius = ceil(round(area_per_shipyard ** (1 / 2)) / 2) + 1
    # print(f'Step {board.step} area_per_shipyard: {area_per_shipyard}, radius {radius}')

    rolling_net_power = player.rolling_net_power
    shipyard_to_scores = defaultdict(list)
    adjusted_kore_arr = player.adjusted_kore_arr[0]

    full_scores = []

    for c in board:
        if c.kore > 200:  # TODO
            continue

        if not shipyards:
            continue

        sorted_shipyards = sorted(shipyards, key=lambda x: x.distance_from(c) - int(x.player_id != player.game_id))
        closest_shipyard = sorted_shipyards[0]
        ship_count_after_convert = max(0, closest_shipyard.available_ships - board.configuration.convert_cost)
        net_power_subzero_idx = np.where(rolling_net_power[:, c.x, c.y] + ship_count_after_convert < 0)[0]

        if len(net_power_subzero_idx) > 0:
            if net_power_subzero_idx.min() <= 8:
                continue
        min_distance = closest_shipyard.distance_from(c)

        if (
                # not closest_shipyard
                closest_shipyard.player_id != player.game_id
                or min_distance < 4
                or min_distance > 6
        ):
            continue

        if len(player.shipyards) >= 2:
            second_closest_shipyard = sorted_shipyards[1]
            second_min_distance = second_closest_shipyard.distance_from(c)
        else:
            second_closest_shipyard = sorted_shipyards[0]
            second_min_distance = second_closest_shipyard.distance_from(c)

        if (
                second_min_distance > 9
                or second_closest_shipyard.player_id != player.game_id
        ):
            continue

        if len(sorted_shipyards) >= 4 and sum([sy.player_id == player.game_id for sy in sorted_shipyards[:4]]) == 4:
            continue

        score = 0
        for p in c.nearby_points_quick(radius):
            distance = p.distance_from(c)
            # adjusted_kore_arr[p.x][p.y]
            score += adjusted_kore_arr[p.x][p.y] * (1 + board.configuration.regen_rate) ** 4 / \
                     (sum(max(0, radius - x.point.distance_from(p)) for x in player.shipyards if distance <= radius) +
                      distance)

        shipyard_to_scores[closest_shipyard].append({"score": score, "point": c})
        full_scores.append({"score": score, "point": c})

    shipyard_targets = []
    for shipyard, scores in shipyard_to_scores.items():
        if scores:
            scores = sorted(scores, key=lambda x: x["score"], reverse=True)
            point = scores[0]["point"]
            shipyard_targets.append((shipyard, point))

    return sorted(full_scores, key=lambda x: x["score"], reverse=True)


def need_more_shipyards(player: Player) -> int:
    board = player.board

    if player.ship_count / (len(player.shipyards) + 1) < 50:
        return 0

    for op in board.players:
        if op != player:
            if (player.ship_count - board.configuration.convert_cost) / (op.ship_count + 1) < 0.55:
                return 0

            if len(player.shipyards) >= len(op.shipyards) * 1.8:
                return 0

            if (player.ship_count - board.configuration.convert_cost) / (len(player.shipyards) + 0.1) / (
                    (op.ship_count) / (len(op.shipyards) + 0.1) + 0.01) < 0.6:
            # if (player.ship_count - board.configuration.convert_cost) / len(player.shipyards) /  ((op.ship_count) / len(op.shipyards)) < 0.6:
                return 0

    fleet_distance = []
    for sy in player.shipyards:
        for f in sy.incoming_allied_fleets:
            fleet_distance.append(len(f.route) * f.final_ship_count)
            # fleet_distance.append(len(f.route))

    if not fleet_distance:
        mean_fleet_distance = 0
    else:
        # mean_fleet_distance = sum(fleet_distance) / len(fleet_distance)
        mean_fleet_distance = sum(fleet_distance) / sum(f.final_ship_count for f in player.fleets)

    shipyard_production_capacity = sum(x.max_ships_to_spawn + 3 for x in player.shipyards)

    steps_left = board.steps_left
    if steps_left > 100:
        scale = 5
    elif steps_left > 50:
        scale = 100
    elif steps_left > 10:
        scale = 100
    else:
        scale = 1000

    expected_kore = player.available_kore()
    for sy in player.shipyards:
        expected_kore += sum(x.expected_kore() for x in sy.incoming_allied_fleets if x.eta <= 6)

    needed = expected_kore > scale * shipyard_production_capacity * min(max(mean_fleet_distance, 5), 10) or (
            player.ship_count / len(player.shipyards)) > 110

    if not needed:
        return 0
    # print(mean_fleet_distance)
    if len(player.shipyards) <= 2 and mean_fleet_distance > 14:
        print(f'step {board.step}, skipped early expansion, mean_fleet_distance={mean_fleet_distance:.1f}')
        return 0

    current_shipyard_count = len(player.shipyards)

    op_shipyard_positions = {
        x.point for x in board.shipyards if x.player_id != player.game_id
    }
    expected_shipyard_count = current_shipyard_count + sum(
        1
        for x in player.fleets
        if x.route.last_action() == Convert or x.route.end in op_shipyard_positions
    )

    opponent_shipyard_count = max(len(x.shipyards) for x in player.opponents)
    opponent_ship_count = max(x.ship_count for x in player.opponents)

    if (
            expected_shipyard_count > opponent_shipyard_count
            and player.ship_count < opponent_ship_count
    ):
        return 0

    if current_shipyard_count < 10:
        if expected_shipyard_count > current_shipyard_count:
            return 0
        else:
            return 1
    return max(0, 2 - (expected_shipyard_count - current_shipyard_count))


def is_safety_route_to_convert(route_points: List[Point], player: Player):
    board = player.board

    target_point = route_points[-1]
    target_time = len(route_points)
    for pl in board.players:
        if pl != player:
            for t, positions in pl.expected_fleets_positions.items():
                if t >= target_time and target_point in positions:
                    return False

    shipyard_positions = {x.point for x in board.shipyards}

    for time, point in enumerate(route_points):
        for pl in board.players:
            is_enemy = pl != player

            if is_enemy:
                if point in shipyard_positions:
                    return False

                if point in pl.expected_fleets_positions[time]:
                    return False

                if point in pl.expected_dmg_positions[time]:
                    return False

    return True
