# from line_profiler_pycharm import profile

# <--->
import numpy as np
from matplotlib import pyplot as plt

from control import is_intercept_direct_attack_route, roundtrip_attack
from geometry import PlanRoute
from mining import mine
from quickboard import Player, Launch, BoardRoute, DoNothing
from helpers import find_shortcut_routes, get_hostile_fleets, absorbables, get_attacked_risks
from logger import logger


# <--->


# @profile
def defend_shipyards(agent: Player):
    board = agent.board
    for pl in board.players:
        if pl != agent:
            op = pl

    need_help_shipyards = []

    for sy in agent.shipyards:

        if sy.action and not isinstance(sy.action, DoNothing):
            continue
        incoming_hostile_fleets = sy.incoming_hostile_fleets
        incoming_allied_fleets = sy.incoming_allied_fleets

        if not incoming_hostile_fleets:
            continue


        sy.incoming_hostile_time = min(x.eta for x in incoming_hostile_fleets)
        # print(set([x.eta for x in incoming_allied_fleets]))
        incoming_hostile_power = sum(x.combined_ship_count for x in incoming_hostile_fleets)
        immediate_incoming_hostile_power = sum(x.combined_ship_count for x in incoming_hostile_fleets if x.eta <= sy.incoming_hostile_time)

        incoming_allied_power = sum(x.ship_count for x in incoming_allied_fleets)
        immediate_incoming_allied_power = sum(
            x.ship_count
            for x in incoming_allied_fleets
            if x.eta <= sy.incoming_hostile_time
        )

        # sy.set_max_route_len(sy.incoming_hostile_time)
        # sy.set_max_mining_distance(sy.incoming_hostile_time // 2)
        # sy.defend_mode = True
        # sy.action_priority = 99

        ships_needed = incoming_hostile_power - incoming_allied_power
        immediate_ships_needed = immediate_incoming_hostile_power - immediate_incoming_allied_power
        # print(immediate_incoming_hostile_power, immediate_incoming_allied_power)
        if sy.ship_count >= immediate_ships_needed:
            sy.set_guard_ship_count(max(0, min(sy.ship_count, int(immediate_ships_needed * 1.1))))
            for f in sy.incoming_hostile_fleets:
                if f.eta <= sy.incoming_hostile_time:
                    f.attacked = True

        elif sy.ship_count < immediate_ships_needed:
            # spawn as much as possible
            immediate_spawn = min(
                int(agent.available_kore() // board.spawn_cost), sy.max_ships_to_spawn)

            resource_share_factor = len(agent.shipyards) * 5

            additional_spawns = min(
                (agent.available_kore() + sum(agent.expected_kore_by_time[:sy.incoming_hostile_time+1])) // board.spawn_cost / resource_share_factor,
                sy.max_ships_to_spawn * sy.incoming_hostile_time
            )

            immediate_spawn = int(max(immediate_spawn, additional_spawns))

            # if (
            #     sy.incoming_hostile_time <= 3
            #     and sum(incoming_allied_fleets[:sy.incoming_hostile_time+1]
            #             ) + sy.ship_count + immediate_spawn + additional_spawns <
            #         sum(incoming_hostile_fleets[:sy.incoming_hostile_time])
            # ):
            #     pass # TODO

            # max_future_ship_counts = max(sy.estimated_ship_counts(
            #     (sy.incoming_hostile_time - sy.expected_build_time, True, False, True))[1:sy.incoming_hostile_time - sy.expected_build_time + 1])

            # eta = sy.incoming_hostile_time
            #
            # if eta <= 2 and (agent.movable_asset < op.movable_asset or agent.ship_count < op.ship_count) and sum(sy.incoming_allied_ships[:3]) <= 1:
            #     # print('xxx')
            #     incoming_hostile_ships = sum(x.final_ship_count for x in sy.incoming_hostile_fleets if x.eta <= eta)
            #     incoming_allied_ships = sum(x.final_ship_count for x in sy.incoming_allied_fleets if x.eta <= eta)
            #
            #     available_to_spawn = min(int(agent.available_kore() // board.spawn_cost), sy.max_ships_to_spawn) + \
            #                          min(int(sum(x.expected_kore() for x in sy.incoming_allied_fleets if
            #                                      x.eta == 1) // board.spawn_cost), sy.max_ships_to_spawn)
            #     if incoming_allied_ships + sy.ship_count + available_to_spawn < incoming_hostile_ships:
            #         print(f'step {board.step}, {sy.point} evacuate')
            #         sy.evacuate = True
            #         mine(agent=agent, shipyards=[sy], use_all_ships=True, must_return=False, ignore_siege=False)
            #         continue

            sy.defend_mode = True
            sy.action_priority = 99
            sy.set_max_route_len(sy.incoming_hostile_time)
            sy.set_max_mining_distance(sy.incoming_hostile_time // 2 - 1)
            sy.set_max_route_len(sy.incoming_hostile_time - 1)
            sy.set_guard_ship_count(max(0, sy.ship_count))

            if immediate_ships_needed > sy.ship_count + immediate_spawn:
                sy.immediate_reinforcement_needs += immediate_ships_needed - (sy.ship_count + immediate_spawn)
                sy.total_reinforcement_needs += ships_needed - (sy.ship_count + immediate_spawn)
                need_help_shipyards.append(sy)

    for sy in need_help_shipyards:
        incoming_hostile_fleets = sy.incoming_hostile_fleets
        incoming_hostile_time = min(x.eta for x in incoming_hostile_fleets)

        other_shipyards = [other_sy for other_sy in agent.shipyards
                           if other_sy != sy and not other_sy.action and other_sy.available_ships]

        immediate_reinforcements = 0
        patience_reinforcements = 0

        reinforcement_spawns = 0
        for other_sy in sorted(other_shipyards, key=lambda x: x.available_ships, reverse=True):

            reinforcement_spawns += max(0, (incoming_hostile_time - other_sy.distance_from(sy))) * other_sy.max_ships_to_spawn

        reinforcement_spawns = min(reinforcement_spawns, agent.available_kore() // board.spawn_cost)

        for other_sy in sorted(other_shipyards, key=lambda x: x.available_ships, reverse=True):
            # if other_sy == sy or other_sy.action or not other_sy.available_ship_count:
            #     continue
            distance = other_sy.distance_from(sy)

            max_current_step_absorbable, max_next_step_absorbable = 0, 0
            for f in agent.fleets:
                current_step_absorbable, next_step_absorbable, _, _ = absorbables(other_sy, f, sy.point)
                if current_step_absorbable and f.ship_count > max_current_step_absorbable:
                    max_current_step_absorbable = f.ship_count
                    other_sy.must_pass = current_step_absorbable
                    # print('current step', f, f.ship_count)
                if next_step_absorbable and f.ship_count > max_next_step_absorbable:
                    max_next_step_absorbable = f.ship_count
                    # print('next step', f, f.ship_count)

            if sy.expected_build_time <= distance <= incoming_hostile_time:
                immediate_reinforcements += other_sy.available_ships + max_current_step_absorbable
                patience_reinforcements += other_sy.available_ships + max_next_step_absorbable

        # print(f'immediate_reinforcements {immediate_reinforcements}, total reinforcement_needs {sy.total_reinforcement_needs},'
        #       f'immediate_reinforcement_needs {sy.immediate_reinforcement_needs}')

        # print(immediate_reinforcements, reinforcement_spawns, sy.immediate_reinforcement_needs)
        if immediate_reinforcements + reinforcement_spawns >= sy.immediate_reinforcement_needs:
            need_reinforcement_spawns = immediate_reinforcements < sy.immediate_reinforcement_needs

            for other_sy in sorted(other_shipyards, key=lambda x: x.available_ships, reverse=True):
                if sy.total_reinforcement_needs <= 0 and sy.immediate_reinforcement_needs <= 0:
                    continue

                if other_sy.distance_from(sy) == incoming_hostile_time or not need_reinforcement_spawns:
                    ships_to_launch = other_sy.available_ships
                    routes = find_shortcut_routes(
                        board, other_sy.point, sy.point, agent, ships_to_launch, must_pass=other_sy.must_pass
                    )
                    if routes:
                        routes.sort(key=lambda r: r.expected_kore_np(board, ships_to_launch), reverse=True)
                        msg = f"Step {board.step} {other_sy} send reinforcements {ships_to_launch} " \
                              f"for shipyard {other_sy.point}->{sy.point}. Must pass： {other_sy.must_pass}"
                        logger.info(msg)
                        print(msg)
                        other_sy.action = Launch(
                            ships_to_launch, routes[0]
                        )

                        sy.immediate_reinforcement_needs -= ships_to_launch
                        sy.total_reinforcement_needs -= ships_to_launch
                        # print(other_sy, ships_to_launch, sy, sy.immediate_reinforcement_needs)

                        other_sy.reinforcement_mode = True
                        other_sy.reinforcement_target = sy
                        # other_sy.mining_destination = [sy.point]
                        # other_sy.max_route_len = incoming_hostile_time - 1

        # else:
        #     for other_sy in sorted(other_shipyards, key=lambda x: x.available_ships, reverse=True):
                else:
                    distance = other_sy.distance_from(sy)

                    if (
                        (distance < incoming_hostile_time or distance < sy.expected_build_time)
                        and other_sy.available_ships < sy.total_reinforcement_needs
                        and agent.available_kore() >= other_sy.max_ships_to_spawn * board.spawn_cost
                    ):
                        # other_sy.action = Spawn(other_sy.max_ships_to_spawn)
                        other_sy.reinforcement_mode = True
                        other_sy.reinforcement_target = sy
                        other_sy.mining_destination = [sy.point]
                        other_sy.max_route_len = incoming_hostile_time - 1
                        # sy.reinforcement_needs -= other_sy.max_ships_to_spawn

                    if (
                        distance < incoming_hostile_time
                        # and patience_reinforcements >= sy.reinforcement_needs
                    ):
                        other_sy.reinforcement_mode = True
                        other_sy.reinforcement_target = sy
                        other_sy.mining_destination = [sy.point]
                        other_sy.max_route_len = incoming_hostile_time - 1


        if sy.immediate_reinforcement_needs > 0 and sy.total_reinforcement_needs > 0:
            eta = sy.incoming_hostile_time
            incoming_hostile_ships = sum(x.final_ship_count for x in sy.incoming_hostile_fleets if x.eta <= eta)
            incoming_allied_ships = sum(x.final_ship_count for x in sy.incoming_allied_fleets if x.eta <= eta)

            available_to_spawn = min(int(agent.available_kore() // board.spawn_cost), sy.max_ships_to_spawn) + \
                                 min(int(sum(x.expected_kore() for x in sy.incoming_allied_fleets if
                                             x.eta == 1) // board.spawn_cost), sy.max_ships_to_spawn)
            if incoming_allied_ships + sy.ship_count + available_to_spawn < incoming_hostile_ships and eta <= 3:
                # sy.last_attempt_attack = True

                incoming_etas = sorted(set([x.eta for x in sy.incoming_hostile_fleets]))

                # only one incoming fleet
                for other_sy in other_shipyards:
                    if other_sy.action:
                        continue
                    if other_sy.distance_from(sy) <= eta or other_sy.distance_from(sy) > 7:
                        continue

                    if (
                        len(incoming_etas) > 1
                        and incoming_etas[1] <= other_sy.distance_from(sy)
                        and other_sy.available_ships < sum(x.final_ship_count for x in sy.incoming_hostile_fleets
                                                           if x.eta == incoming_etas[1]) + sy.immediate_reinforcement_needs + (other_sy.distance_from(sy)-eta)*2
                    ):
                        continue

                    op_reinforcements = 0
                    for op_sy in op.shipyards:
                        if op_sy.distance_from(sy) <= other_sy.distance_from(sy):
                            op_reinforcements += op_sy.available_ships + (op_sy.distance_from(sy) - other_sy.distance_from(sy))*op_sy.max_ships_to_spawn

                    if other_sy.available_ships > sy.immediate_reinforcement_needs + (other_sy.distance_from(sy)-eta)*2 + op_reinforcements:
                        ships_to_launch = other_sy.available_ships
                        routes = find_shortcut_routes(
                            board, other_sy.point, sy.point, agent, ships_to_launch
                        )
                        if routes:
                            routes.sort(key=lambda r: r.expected_kore_np(board, ships_to_launch), reverse=True)
                            msg = f"Step {board.step} {other_sy} send reinforcements {ships_to_launch} " \
                                  f"for recapturing shipyard {other_sy.point}->{sy.point}. "
                            logger.info(msg)
                            print(msg)
                            other_sy.action = Launch(
                                ships_to_launch, routes[0]
                            )
                            sy.recaptured = True
                            break
                            # sy.immediate_reinforcement_needs -= ships_to_launch

                            # other_sy.reinforcement_mode = True
                            # other_sy.reinforcement_target = sy
                            # other_sy.mining_destination = [sy.point]
                            # other_sy.max_route_len = incoming_hostile_time - 1


def joint_defend_fleets(agent: Player):
    board = agent.board

    if not agent.shipyards:
        return

    opponent_shipyard_points = {x.point for x in board.shipyards if x.player_id != agent.game_id}

    agent_fleets = sorted([f for f in agent.fleets if not f.suicide_attacker], key=lambda f: f.expected_kore(), reverse=True)
    for f in agent_fleets:
        # time_to_remaining_health = fleet_remaining_health(f, agent)
        time_to_remaining_health = f.route_health
        if not time_to_remaining_health:
            continue

        # attacked_risks = get_attacked_risks(agent, f)
        # for idx, p_h in enumerate(time_to_remaining_health, 1):
        #     if idx < len(attacked_risks):
        #         p_h['health'] -= attacked_risks[idx]

        if all(t['health'] > 0 for t in time_to_remaining_health) or f.absorbed:
            continue

        need_rescued = True
        ships_needed = -min(t['health'] for t in time_to_remaining_health)
        print(f"Step {board.step} {f.ship_count} ships @ {f.point} being attacked, need {ships_needed} ships")

        joint_reinforcements = []
        remaining_ships_last_step = time_to_remaining_health[0]['health']

        shipyards = agent.shipyards.copy()
        for time, point_health_dict in enumerate(time_to_remaining_health, 1):
            if ships_needed < 0:
                break

            point = point_health_dict['point']
            if remaining_ships_last_step <= 0:
                break

            if remaining_ships_last_step > 0:

                shipyards_in_range = sorted([sy for sy in shipyards if sy.point.distance_from(point) == time
                                             and not sy.action],
                                            key=lambda sy: sy.available_ships,
                                            reverse=True)

                for sy in shipyards_in_range:
                    joint_reinforcements.append({'distance': time,
                                                 'shipyard': sy,
                                                 'point': point_health_dict['point'],
                                                 'health': point_health_dict['health'],
                                                 'last_health': remaining_ships_last_step,
                                                 'ships': sy.available_ships})
                    shipyards.remove(sy)

                if sum(reinforcement['ships'] for reinforcement in joint_reinforcements if
                       min(reinforcement['ships'], ships_needed) < reinforcement['last_health']) >= ships_needed:

                    for reinforcement in sorted(joint_reinforcements, key=lambda x: x['ships'], reverse=True):
                        if min(reinforcement['ships'], ships_needed) >= reinforcement['last_health']:
                            continue

                        if ships_needed < 0:
                            break

                        num_ships_to_launch = int(min(reinforcement['shipyard'].available_ships,
                                                  max(10, int(ships_needed * 1.2) + 1),
                                                  reinforcement['last_health']))

                        sy = reinforcement['shipyard']
                        target_point = reinforcement['point']
                        health = reinforcement['health']

                        # TODO routes permutations
                        routes = find_shortcut_routes(
                            board, sy.point, target_point, agent, num_ships_to_launch
                        )

                        if routes:
                            msg = f"Step {board.step} Send {num_ships_to_launch} reinforcements to join fleet " \
                                  f"{sy.point}--{num_ships_to_launch}-->{f}@{target_point}"
                            print(msg)
                            logger.info(msg)
                            routes.sort(key=lambda r: r.expected_kore_np(board, num_ships_to_launch), reverse=True)
                            sy.action = Launch(
                                num_ships_to_launch, routes[0]
                            )
                            ships_needed -= num_ships_to_launch

                else:
                    for reinforcement in sorted(joint_reinforcements, key=lambda x: x['ships'], reverse=True):
                        if reinforcement['ships'] < reinforcement['last_health'] or (reinforcement['health'] < 0 and reinforcement['ships'] < ships_needed):
                            # print('continue')
                            continue

                        num_ships_to_launch = int(min(reinforcement['shipyard'].available_ships,
                                                  max(3, int(ships_needed * 1.2))))

                        sy = reinforcement['shipyard']
                        target_point = reinforcement['point']
                        health = reinforcement['health']

                        out_paths = sy.point.dirs_to(target_point)
                        out_plans = PlanRoute(out_paths).permutations()

                        destinations = sorted(agent.shipyards, key=lambda x: x.point.distance_from(target_point))
                        destination = destinations[0].point
                        # destination = point_to_closest_shipyard[point]  # return destination for the attacking fleet

                        back_paths = target_point.dirs_to(destination)
                        back_plans = PlanRoute(back_paths).permutations()
                        complete_plans = [out_plan + back_plan for out_plan in out_plans
                                          for back_plan in back_plans]

                        for plan in complete_plans:
                            if num_ships_to_launch < plan.min_fleet_size():
                                continue

                            route = BoardRoute(sy.point, plan)

                            if any(x in opponent_shipyard_points for x in route.points()):
                                continue

                            if is_intercept_direct_attack_route(route, agent, direct_attack_fleet=f):
                                continue

                            msg = f"Step {board.step} Send {num_ships_to_launch} reinforcements to absorb " \
                                  f"fleet {sy.point}--{num_ships_to_launch}->{target_point}, distance={time}"
                            logger.info(msg)
                            print(msg)
                            sy.action = Launch(num_ships_to_launch, route)
                            ships_needed = -1
                            break

                # break
            remaining_ships_last_step = point_health_dict['health']

        if need_rescued and ships_needed > 0:
            hostile_fleets = get_hostile_fleets(f, agent)
            roundtrip_attack(agent=agent, targets=hostile_fleets, minimum_damage=ships_needed, action_type='Rescue')

