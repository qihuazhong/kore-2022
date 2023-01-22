import random

import numpy as np
from typing import List, Optional
# from line_profiler_pycharm import profile
import itertools
from matplotlib import pyplot as plt
# <--->
from basic import min_ship_count_for_flight_plan_len
from geometry import Point
from quickboard import Player, Launch, Shipyard, MiningRoute, SuicideRoute, Spawn
from helpers import is_intercept_route, find_non_intercept_routes, route_danger_sparse


# <--->

# @profile
def mine(agent: Player, default_max_distance: Optional[int] = None, default_max_route_len: int = 26,
         shipyards: Optional[List[Shipyard]] = None,
         use_all_ships=False, must_return: bool = False, ignore_siege=False,
         roundtrip_attack_mode=False, must_pass=None, roundtrip_attack_target=None, absorbed_ships=0,
         ignore_risk=False, last_attempt=False, rescue=False):
    """
    """
    board = agent.board
    if not agent.opponents:
        return

    mining_sy = []

    safety = False
    my_ship_count = agent.ship_count
    op_ship_count = max(x.ship_count for x in agent.opponents)
    if my_ship_count < 2 * op_ship_count:
        safety = True

    op_ship_count = []
    for op in agent.opponents:
        for fleet in op.fleets:
            op_ship_count.append(fleet.ship_count)

    shipyard_count = len(agent.shipyards)
    if not default_max_distance:
        if shipyard_count < 3:
            default_max_distance = 13
        elif shipyard_count < 10:
            default_max_distance = 12
        elif shipyard_count < 15:
            default_max_distance = 11
        elif shipyard_count < 20:
            default_max_distance = 10
        else:
            default_max_distance = 7

    default_max_distance = min(int(board.steps_left // 2), default_max_distance)
    default_max_route_len = min(board.steps_left, default_max_route_len)

    if not shipyards:
        shipyards = [sy for sy in agent.shipyards if not sy.action]

    expected_kore = agent.available_kore()
    for sy in shipyards:
        expected_kore += sum(x.expected_kore() for x in sy.incoming_allied_fleets if x.eta <= 20)

    mining_shipyards = [sy for sy in shipyards if not sy.action and sy.ship_count >= 2]
    # max_routes_per_point = 100 + len(mining_shipyards)

    if rescue:
        horizon = max(time for time, point in must_pass) - 1
    else:
        horizon = 21

    for sy in shipyards:
        # if sy.action and not sy.reinforcement_mode:
        #         continue
        if sy.reinforcement_mode and agent.available_kore() >= sy.max_ships_to_spawn * board.spawn_cost:
            continue

        must_return = False
        # use_all_ships = default_use_all_ships
        ignore_siege = False
        max_route_len = default_max_route_len
        max_distance = default_max_distance

        if sy.reinforcement_mode:
            use_all_ships = True
            ignore_siege = True
            max_route_len = sy.max_route_len

        if sy.defend_mode:
            if sy.last_attempt_attack:
                must_return = False
                use_all_ships = True
                ignore_siege = False
            elif not sy.evacuate:
                must_return = True
                use_all_ships = True
                ignore_siege = True
                max_distance = sy.max_mining_distance
                max_route_len = sy.max_route_len
            else:
                must_return = False
                use_all_ships = True
                ignore_siege = False

        if use_all_ships:
            free_ships = sy.ship_count
        else:
            if expected_kore < min(board.spawn_cost * sy.max_ships_to_spawn, 50):
                free_ships = sy.available_ships
                max_distance = max_distance // 2  # max distance override
            else:
                free_ships = sy.available_ships
        if (
                free_ships < 2 or
                (free_ships == 2 and agent.kore >= board.spawn_cost)
        ):
            continue

        num_ships_to_launch = free_ships
        next_step_ships_shortage = num_ships_to_launch >= 22 and sum(sy.incoming_allied_ships[1:3]) < 42

        max_num_ships_to_launch = free_ships + sy.incoming_allied_ships[1] + min(
            sy.max_ships_to_spawn, agent.available_kore() // board.spawn_cost)

        feasible_route = None
        destinations_count = 0
        max_destinations_count = 0
        # if roundtrip_attack_mode:
        #     max_destinations_count += 1
        while feasible_route is None and destinations_count <= max_destinations_count:

            routes = find_zig_zag_shipyard_mining_routes(agent=agent, sy=sy, num_ships=max_num_ships_to_launch,
                                                         destinations=sy.mining_destination, safety=safety,
                                                         max_distance=max_distance, max_route_len=max_route_len,
                                                         must_return=must_return, ignore_siege=ignore_siege,
                                                         roundtrip_attack_mode=roundtrip_attack_mode, must_pass=must_pass,
                                                         absorbed_ships=absorbed_ships,
                                                         next_step_ships_shortage=next_step_ships_shortage,
                                                         last_attempt=last_attempt,
                                                         destinations_count = destinations_count
                                                         )
            destinations_count += 1

            route_to_score = {}

            for route in routes:
                route_to_score[route] = route.expected_kore_sparse(agent, num_ships_to_launch) * \
                        board.accum_discount_rate((agent, len(route),)) / len(route)

            if not route_to_score:
                continue

            routes = sorted(route_to_score, key=lambda r: -route_to_score[r])
            feasible_route = None
            potential_next_route = None
            next_route_with_patience = None
            scaled_down_potential_route = None

            next_launch_ships_with_patience = num_ships_to_launch + sy.incoming_allied_ships[1] + min(
                sy.max_ships_to_spawn, agent.available_kore() // board.spawn_cost)

            # next_route_with_patience_2 = None
            # next_launch_ships_with_patience_2 = num_ships_to_launch + sum(sy.incoming_allied_ships[1:3]) + min(
            #     sy.max_ships_to_spawn, agent.available_kore() // board.spawn_cost)

            power_opti, power_pessi = agent.net_power((6, sy))
            if roundtrip_attack_target:
                pad_right = max(0, 22 - 1 - roundtrip_attack_target.expected_dmg_positions_np.shape[0])
                compensate = np.concatenate([np.zeros(shape=(1, board.size, board.size)),
                                                  roundtrip_attack_target.expected_dmg_positions_np,
                                                  np.zeros(shape=(pad_right, board.size, board.size))])[:22]
                power_pessi += compensate
            if roundtrip_attack_target and roundtrip_attack_target.route_health:
                t = min(must_pass[0][0]-2, len(roundtrip_attack_target.route_health)-1)
                target_fleet_ship_count = roundtrip_attack_target.route_health[t]['health']
            else:
                target_fleet_ship_count = 0

            # print(sy, roundtrip_attack_mode, len(routes), must_pass)
            for route in routes:
                if must_pass is not None:
                    is_continue = False
                    for time, point in must_pass:
                        _, rows, cols = route.sparse_rep
                        if time > len(route) or not (rows[time-1] == point.x and cols[time-1] == point.y):
                            is_continue = True

                    if is_continue:
                        continue
                # num_ships_to_launch = min(free_ships, max_fleet_size)

                route_danger_opti = route_danger_sparse(board, route, power_opti, horizon=horizon)
                route_danger_pessi = route_danger_sparse(board, route, power_pessi, horizon=horizon)

                ships_surplus_opti = route_danger_opti + num_ships_to_launch + absorbed_ships
                ships_surplus_pessi = route_danger_pessi + num_ships_to_launch - target_fleet_ship_count*(not rescue) + absorbed_ships

                if (
                    next_route_with_patience is None
                    and route_danger_opti + next_launch_ships_with_patience >= 0
                    and next_launch_ships_with_patience >= route.min_fleet_size
                ):
                    next_route_with_patience = route
                    route_danger_opti_with_patience = route_danger_opti

                if num_ships_to_launch < route.min_fleet_size:
                    continue
                elif ships_surplus_opti <= 0:
                    continue
                else:
                    if feasible_route is None and (ships_surplus_pessi >= 0 or ignore_risk):
                        feasible_route = route
                        surplus = ships_surplus_pessi

                    if next_step_ships_shortage:
                        next_launch_ships_split3 = (num_ships_to_launch + sum(sy.incoming_allied_ships[1:3])) // 3
                        next_launch_ships_split2 = (num_ships_to_launch + sy.incoming_allied_ships[1]) // 2

                        if potential_next_route is None and route_danger_opti + next_launch_ships_split3 > 0 and next_launch_ships_split3 >= route.min_fleet_size:
                            potential_next_route = route
                            surplus_split_3 = route_danger_pessi + next_launch_ships_split3

                        if scaled_down_potential_route is None and route_danger_pessi + next_launch_ships_split2 > 0 and next_launch_ships_split2 >= route.min_fleet_size:
                            scaled_down_potential_route = route
                            surplus_split_2 = route_danger_pessi + next_launch_ships_split2

                    if feasible_route and potential_next_route and scaled_down_potential_route:
                        break

        if feasible_route is None or route_to_score[feasible_route] <= 0:
            continue

        if not roundtrip_attack_mode \
            and route_to_score[next_route_with_patience] / route_to_score[feasible_route] > 1.4 \
            and sy.incoming_allied_ships[1] + agent.available_kore() // board.spawn_cost > 0: # and not sy.is_supply_depot:
            print(f'Step {board.step} {sy.point} wait for next step to mine. Scores: {route_to_score[next_route_with_patience]:.0f} /'
                  f'{route_to_score[feasible_route]:.0f} ')
            # print(route_danger_opti_with_patience)
            num_to_spawn = int(min(agent.available_kore() // board.spawn_cost, sy.max_ships_to_spawn))
            if num_to_spawn > 0:
                sy.action = Spawn(num_to_spawn)
            continue

        # scale down
        if (
            next_step_ships_shortage
            and not sy.evacuate
            and not sy.reinforcement_mode
            and not use_all_ships
            and not roundtrip_attack_mode
            and board.step > 5
            and scaled_down_potential_route is not None
            and route_to_score[scaled_down_potential_route] / route_to_score[feasible_route] > 0.75
        ):
            if sy.is_supply_depot:
                divide_factor = 2
            else:
                divide_factor = 3

            chosen_route = scaled_down_potential_route
            num_ships_to_launch = int(next_launch_ships_split2)
            num_ships_to_launch = int(min(sy.available_ships,
                                          max(scaled_down_potential_route.min_fleet_size,
                                          (sy.available_ships + sum(sy.incoming_allied_ships[1:3])) // divide_factor,
                                          num_ships_to_launch - max(1, surplus_split_2) + 3)))
            # print(scaled_down_potential_route.min_fleet_size,
            #                               (sy.available_ships + sum(sy.incoming_allied_ships[1:3])) // divide_factor,
            #                               num_ships_to_launch - max(1, surplus_split_2) + 3)

        else:
            chosen_route = feasible_route
            if (
                sy.incoming_allied_ships[1] < feasible_route.min_fleet_size
                and sy.available_ships - feasible_route.min_fleet_size + sy.incoming_allied_ships[1] < num_ships_to_launch
                and num_ships_to_launch > 5
                and not sy.is_supply_depot
                and not sy.evacuate
                and not use_all_ships
                and board.step > 5
                # and not roundtrip_attack_mode
            ):
                # print(
                #         feasible_route.min_fleet_size,
                #             (sy.available_ships + sum(sy.incoming_allied_ships[1:3])) // 3,
                #             num_ships_to_launch - max(1, surplus) + 3
                # )
                num_ships_to_launch = int(min(sy.available_ships,
                                                max(feasible_route.min_fleet_size,
                                                    (sy.available_ships + sum(sy.incoming_allied_ships[1:3])) // 3,
                                                    num_ships_to_launch - max(1, surplus) + 3,
                                                    absorbed_ships+1
                                                    )))

        # print(sy, route_to_score[chosen_route], num_ships_to_launch)

        sy.action = Launch(num_ships_to_launch, chosen_route)
        agent.efficiency_tracker.update(sy, agent, chosen_route)
        mining_sy.append(sy)

        continue

    return mining_sy


def find_suicide_route(agent: Player, departure: Point, num_ships: int,
                       target_point: Point, route_len: int, max_sustained_dmg: int = 0
                       ) -> List:
    routes_candidate = agent.preloaded_mining_routes[
        ((target_point.x - departure.x) % agent.board.size, (target_point.y - departure.y) % agent.board.size)]

    routes_candidate = list(itertools.chain(*[r for com_len, r in routes_candidate.items() if
                                              num_ships >= min_ship_count_for_flight_plan_len(com_len-1)]))

    routes_candidate = [r for r in routes_candidate
                        if r['len'] == route_len
                        and num_ships >= min_ship_count_for_flight_plan_len(r['oneway_com_len'])]

    commands = routes_candidate

    routes = []
    for command in commands:
        xy_pairs = ((departure.x, departure.y) + command['path']) % agent.board.size
        route = SuicideRoute(
            departure_x=departure.x,
            departure_y=departure.y,
            xy_pairs=xy_pairs)

        if not is_intercept_route(route, agent, safety=True, allow_shipyard_intercept=False, allow_friendly_fleet=True,
                                  max_sustained_dmg=max_sustained_dmg):
            routes.append(route)

    return routes


def mining_target_time_limit(sy: Shipyard, power, mining_target_points: List[Point], num_ships: int):

    limit_dict = {}
    for c in mining_target_points:

        first_neg_index = np.argmax((power[:, c.x, c.y]+num_ships < 0))
        if first_neg_index == 0:
            first_neg_index = 26
        limit_dict[c] = first_neg_index


    return limit_dict


# @profile
def find_zig_zag_shipyard_mining_routes(agent: Player, sy: Shipyard, num_ships: int, destinations: List[Point] = None,
                                        safety=True, max_distance: int = 15, max_route_len: Optional[int] = None,
                                        must_return: bool = False, ignore_siege=False, roundtrip_attack_mode=False,
                                        must_pass=None, absorbed_ships=0, next_step_ships_shortage=False,
                                        last_attempt=False, destinations_count : int = 0) -> List:

    if max_distance < 1:
        return []

    power_opti, power_pessi = agent.net_power((6, sy))

    if next_step_ships_shortage:
        power = power_opti
    else:
        power = power_pessi

    departure = sy.point

    future_destinations = {}
    point_to_shipyard = {}
    for shipyard in sy.player.shipyards:
        point_to_shipyard[shipyard.point] = shipyard
        if shipyard.expected_build_time > 0:
            future_destinations[shipyard.point] = shipyard.expected_build_time

    if destinations is None:
        destinations = set()
        for shipyard in sy.player.shipyards:
            siege = sum(x.ship_count for x in shipyard.incoming_hostile_fleets)
            if siege > shipyard.ship_count and not ignore_siege:
                continue
            # if shipyard.is_supply_depot:
            #     continue
            if shipyard.evacuate:
                continue

            if last_attempt and shipyard.last_attempt_attack:
                continue
            destinations.add(shipyard.point)

        if not destinations:
            return []

    high_kore_threshold = np.percentile([p.kore for p in sy.point.nearby_points_quick(max_distance)], 85)

    if not sy.reinforcement_mode and not roundtrip_attack_mode:  # and not sy.is_supply_depot:
        skipped_points = set([p for p in sy.point.nearby_points_quick(max_distance)
                          if max([p.kore] + [adj_p.kore for adj_p in p.adjacent_points]) < high_kore_threshold
                          or p in agent.board.low_kore_points])
    else:
        skipped_points = []

    routes = []

    mining_target_points = sy.point.nearby_points_quick(max_distance)
    mining_target_points = [c for c in mining_target_points if
                            c != departure
                            and c not in destinations
                            and c not in skipped_points
                            # and c not in agent.board.low_kore_points
                            ]

    if roundtrip_attack_mode:
        must_pass_t, must_pass_p = must_pass[0]
        mining_target_points = [c for c in mining_target_points if c in must_pass_p.nearby_points_quick(5) + [must_pass_p]]

    if not mining_target_points:
        print(f'warning !!! {sy} has not mining target points')

    limit_dict = mining_target_time_limit(sy, power, mining_target_points + list(destinations), num_ships+absorbed_ships)
    # print(limit_dict)

    if must_return and destinations_count == 1:
        return []
    if len(destinations) <= destinations_count:
        return []

    if destinations_count==1:
        print(f'Step {agent.board.step} considering second destinations {num_ships:.0f} ships for {sy} @{sy.point}')

    for c in mining_target_points:

        if must_return:
            destination = departure
        else:
            destination = sorted(destinations, key=lambda x: c.distance_from(x) + point_to_shipyard[x].takeover_risk)[destinations_count]

        # TODO consider the top 2 destinations
        out_routes = agent.preloaded_mining_routes[
            ((c.x - departure.x) % agent.board.size, (c.y - departure.y) % agent.board.size)]

        return_routes = agent.preloaded_return_routes[
            ((destination.x - c.x) % agent.board.size, (destination.y - c.y) % agent.board.size)]

        if num_ships < 21: # and not sy.is_supply_depot:
            out_routes = list(itertools.chain(*[r for com_len, r in out_routes.items() if com_len <= 3]))
        elif num_ships < 34:
            out_routes = list(itertools.chain(*[r for com_len, r in out_routes.items() if com_len <= 4]))
        elif num_ships < 55:
            out_routes = list(itertools.chain(*[r for com_len, r in out_routes.items() if com_len <= 5]))
        else:
            out_routes = list(itertools.chain(*[r for com_len, r in out_routes.items()]))

        out_routes = [r for r in out_routes if r['distance'] < limit_dict[c]]

        if max_route_len:
            out_len = max_route_len - c.distance_from(destination)
            out_routes = [r for r in out_routes if r['len'] <= out_len]

            if agent.board.step == 5:
                out_routes = [r for r in out_routes if r['len'] >= 10 - c.distance_from(destination)]

            # if point_to_shipyard[destination].take_over_time <= 12:
            #     out_routes = [r for r in out_routes if r['len'] >= point_to_shipyard[destination].take_over_time - c.distance_from(destination)]
            #     print(f'step {agent.board.step}, warning, limiting mining len @ {point_to_shipyard[destination].take_over_time} steps to {destination}')

        if not out_routes:
            commands = []
        else:
            sample_num = int(150 - max(1, len(agent.shipyards) / 20) * 70 - sy.is_supply_depot * 15)
            commands = random.sample(out_routes, k=min(sample_num, len(out_routes)))

        route_candidates = []
        return_xy_pairs = [((c.x, c.y) + return_routes[i]['path']) for i in range(len(return_routes))]
        return_commands = [return_routes[i]['command'] for i in range(len(return_routes))]

        for idx, command in enumerate(commands):

            hash_key = (command['command'], departure.x, departure.y, destination.x, destination.y)

            route_candidates += (
                agent.route_cache.get_route(
                    path=command['path'],
                    return_xy_pairs=return_xy_pairs,
                    hash_key=hash_key,
                    return_commands=return_commands
                 ))

        if destination in future_destinations.keys():
            route_candidates = [r for r in route_candidates if len(r) >= future_destinations[destination]]

        if sy.evacuate:
            _len = destination.distance_from(departure)
            route_candidates = [r for r in route_candidates if len(r) <= _len]

        routes += route_candidates

    routes = list(set(routes))

    if not roundtrip_attack_mode:
        intercepts = find_non_intercept_routes(routes, agent, safety)
        routes = np.array(routes)[intercepts]
    # else:
    #     intercepts = find_non_intercept_routes(routes, agent, safety, exclude=must_pass)
    #     print(must_pass)
    #     routes = np.array(routes)[intercepts]

    return routes
