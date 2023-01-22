import math
from collections import defaultdict

# from line_profiler_pycharm import profile

# <--->
from typing import Optional, List, DefaultDict

import numpy as np
from matplotlib import pyplot as plt

from geometry import PlanRoute
from mining import find_suicide_route, mine
from quickboard import Player, Launch, Spawn, Fleet, FleetPointer, BoardRoute, Shipyard, AttackOpportunity
from helpers import is_invitable_victory, find_shortcut_routes, is_protected_route, is_better_than_mining, absorbables
from logger import logger


# <--->


# @profile
def roundtrip_attack(agent: Player, max_distance: Optional[int] = None, cap_launched_ships=False, targets=None,
                     minimum_damage=None, action_type='Roundtrip attack', max_patience=7):
    board = agent.board

    if board.step <= 400:
        cap_launched_ships = True

    if agent.movable_asset > sum(op.movable_asset for op in agent.opponents):
        max_patience += 1

    shipyard_count = len(board.shipyards)
    if not max_distance:
        if shipyard_count < 8:
            max_distance = 10
        elif shipyard_count < 14:
            max_distance = 8
        else:
            max_distance = 8
    max_distance = min(board.steps_left, max_distance)

    for pl in agent.opponents:
        if pl != agent:
            op = pl

    op_ship_count = []
    # for op in agent.opponents:
    for fleet in op.fleets:
        op_ship_count.append(fleet.ship_count)
    for sy in op.shipyards:
        if sy.ship_count > 2:
            op_ship_count.append(sy.ship_count)

    if not op_ship_count:
        mean_fleet_size = 0
        max_fleet_size = np.inf
    else:
        mean_fleet_size = np.percentile(op_ship_count, 80)
        max_fleet_size = math.ceil(max(op_ship_count) * 1.1)

    if targets is None:
        targets = []
        for x in agent.opponents:
            for fleet in x.fleets:
                targets.append(fleet)

    if not targets:
        return

    targets.sort(key=lambda f: f.loot_value, reverse=True)
    shipyards = [
        x for x in agent.shipyards if x.ship_count > 0
                                      and (not x.action or x.solo_capturer)
                                      and not x.reinforcement_mode
        # TODO allow sy in reinforcement mode to attack incomings
    ]
    if not shipyards:
        return

    point_to_closest_shipyard = {}
    for p in board:
        closest_shipyard = None
        min_distance = board.size
        for sy in agent.shipyards:
            distance = sy.point.distance_from(p)
            if distance < min_distance:
                min_distance = distance
                closest_shipyard = sy
        point_to_closest_shipyard[p] = closest_shipyard.point

    opponent_shipyard_points = {x.point for x in board.shipyards if x.player_id != agent.game_id}

    latent_attack_opportunities: DefaultDict[Fleet, List[AttackOpportunity]] = defaultdict(list)
    immediate_attack_opportunities: DefaultDict[Fleet, List[AttackOpportunity]] = defaultdict(list)
    patience_attack_opportunities: DefaultDict[Fleet, List[AttackOpportunity]] = defaultdict(list)

    for target_fleet in targets:

        if target_fleet.final_ship_count <= 0 and not target_fleet.suicide_attacker:
            continue

        if minimum_damage:
            min_ships_to_send = minimum_damage
        else:
            min_ships_to_send = target_fleet.final_ship_count + 1  # TODO

        for sy in shipyards:
            available_ship_count = sy.available_ships
            if sy.defend_mode and sy.total_reinforcement_needs > 0:
                eta = sy.incoming_hostile_time
                incoming_hostile_ships = sum(x.final_ship_count for x in sy.incoming_hostile_fleets if x.eta <= eta)
                incoming_allied_ships = sum(x.final_ship_count for x in sy.incoming_allied_fleets if x.eta <= eta)

                available_to_spawn = min(int(agent.available_kore() // board.spawn_cost), sy.max_ships_to_spawn) + \
                                     min(int(sum(x.expected_kore() for x in sy.incoming_allied_fleets if
                                                 x.eta == 1) // board.spawn_cost), sy.max_ships_to_spawn)
                if incoming_allied_ships + sy.ship_count + available_to_spawn < incoming_hostile_ships and eta <= 2:
                    if target_fleet in sy.incoming_hostile_fleets and not sy.recaptured:
                        available_ship_count = sy.ship_count
                        sy.last_attempt_attack = True

            if target_fleet.attacked:
                break

            # if sy.action or available_ship_count + sy.incoming_allied_ships[1:max_patience + 1].sum() < min_ships_to_send:
            #     continue

            # if sy.action and not sy.solo_capturer:
            #     continue

            if sy.solo_capturer:
                if len(agent.shipyards) < len(op.shipyards) + 1:
                    continue
                if target_fleet.expected_kore() < 150:
                    continue

            if cap_launched_ships and not sy.last_attempt_attack:
                num_ships_to_launch = min(available_ship_count, max(max_fleet_size, int(min_ships_to_send * 1.1)))
            else:
                num_ships_to_launch = available_ship_count

            for target_time, target_ship_point in enumerate(target_fleet.route.points(), 1):
                if not target_fleet.suicide_attacker and target_time == len(target_fleet.route):
                    # Only attack at the last point if it is a suicide attacker
                    continue
                elif target_fleet.suicide_attacker and target_time == len(target_fleet.route):
                    # If it is a suicide attacker, only attack by collision
                    # TODO only attack by collision if there is already adj damage
                    target_points = [target_ship_point]
                else:
                    target_points = [target_ship_point] + target_ship_point.adjacent_points

                if target_fleet.attacked:
                    break

                if target_time > max_distance:
                    continue

                latent_target_points = [p for p in target_points if sy.point.distance_from(p) < target_time]
                for latent_target_point in latent_target_points:
                    if sy.incoming_allied_ships[
                        target_time - sy.point.distance_from(latent_target_point)
                    ] >= min_ships_to_send:
                        latent_attack_opportunities[target_fleet].append(
                            AttackOpportunity(departure=sy,
                                              target_time=target_time,
                                              target_point=latent_target_point,
                                              num_ships_to_launch=num_ships_to_launch,
                                              collision=latent_target_point == target_ship_point)
                        )

                immediate_target_points = [p for p in target_points if sy.point.distance_from(p) == target_time]
                for immediate_target_point in immediate_target_points:

                    max_current_step_absorbable, max_next_step_absorbable = 0, 0
                    potential_absorbed = 0

                    for f in agent.fleets:
                        if target_fleet in f.damage_to:
                            continue

                        current_step_absorbable, _, _, _ = absorbables(sy, f, immediate_target_point)
                        if current_step_absorbable and f.ship_count > max_current_step_absorbable and \
                                current_step_absorbable[0][0] <= target_time:
                            max_next_step_absorbable = f.ship_count
                            must_pass = current_step_absorbable
                            potential_absorbed = max_next_step_absorbable
                    if available_ship_count + potential_absorbed >= min_ships_to_send or target_fleet.suicide_attacker:

                        if potential_absorbed > 0 and available_ship_count < min_ships_to_send:
                            must_pass = must_pass
                            absorbed_ships = potential_absorbed
                        else:
                            must_pass = None
                            absorbed_ships = 0

                        partial = target_fleet.suicide_attacker and available_ship_count + potential_absorbed < min_ships_to_send
                        immediate_attack_opportunities[target_fleet].append(
                            AttackOpportunity(departure=sy,
                                              target_time=target_time,
                                              target_point=immediate_target_point,
                                              num_ships_to_launch=num_ships_to_launch,
                                              collision=immediate_target_point == target_ship_point,
                                              must_pass=must_pass,
                                              absorbed_ships=absorbed_ships,
                                              partial=partial,
                                              rescue=action_type == 'Rescue'
                                              )
                        )

                for patience in range(1, max_patience + 1):
                    patience_target_points = [p for p in target_points if
                                              sy.point.distance_from(p) == target_time - patience]
                    for patience_target_point in patience_target_points:

                        potential_absorbed = 0
                        _, max_next_step_absorbable = 0, 0
                        if 1 <= patience <= 3:

                            for f in agent.fleets:
                                if target_fleet in f.damage_to:
                                    continue

                                _, next_step_absorbable, step_2, step_3 = absorbables(sy, f, patience_target_point)

                                latent_absorbables = [next_step_absorbable, step_2, step_3]
                                # print(sy, latent_absorbables)
                                if latent_absorbables[patience-1] and f.ship_count > max_next_step_absorbable and \
                                        latent_absorbables[patience-1][0][0] <= target_time:
                                    max_next_step_absorbable = f.ship_count
                                    must_pass = latent_absorbables[patience-1]
                                    potential_absorbed = max_next_step_absorbable

                        future_incomings = sy.incoming_allied_ships[1:patience + 1].sum()
                        if (
                            available_ship_count + future_incomings + potential_absorbed >= min_ships_to_send
                            and min_ships_to_send > future_incomings
                        ):
                            # if the second condition is not True, there is no need keep the current available ships
                            patience_attack_opportunities[target_fleet].append(
                                AttackOpportunity(departure=sy,
                                                  target_time=target_time,
                                                  target_point=patience_target_point,
                                                  num_ships_to_launch=num_ships_to_launch,
                                                  collision=patience_target_point == target_ship_point,
                                                  patience=patience, absorbed_ships=potential_absorbed)
                            )
                            break

    shipyard_future_tasks = defaultdict(lambda: defaultdict(set))  # shipyard -> time- > target fleet
    for target_fleet, attack_opportunities in latent_attack_opportunities.items():
        for ao in attack_opportunities:
            shipyard_future_tasks[ao.departure][ao.target_time].add(target_fleet)

    for target_fleet, attack_opportunities in immediate_attack_opportunities.items():
        immediate_target_times = set(
            ao.target_time for ao in attack_opportunities if not ao.departure.action)

        patience_collision_opportunity = False
        for pao in patience_attack_opportunities[target_fleet]:
            if pao.collision:
                patience_collision_opportunity = True

        if len(immediate_target_times) == 0:
            continue

        if min(immediate_target_times) > 3:

            if target_fleet in latent_attack_opportunities and action_type != 'Rescue':
                for ao in latent_attack_opportunities[target_fleet]:
                    if len(shipyard_future_tasks[ao.departure][ao.target_time]) < 2:
                        print(f'Step {board.step} Delay attack {target_fleet.ship_count} ships @{target_fleet.point} ')
                        target_fleet.latent_target = True

                    if not ao.collision and any(immediate_ao.collision for immediate_ao in attack_opportunities):
                        target_fleet.latent_target = False

                    if target_fleet.latent_target:
                        break

            if not target_fleet.latent_target:
                for sy in shipyard_future_tasks:
                    for other_target_time in shipyard_future_tasks[sy]:
                        shipyard_future_tasks[sy][other_target_time].discard(target_fleet)

        if target_fleet.latent_target:
            continue

        if len(immediate_target_times) == 0:
            break
        else:
            attack_opportunities.sort(key=lambda ao: (~ao.partial, ao.target_time, ~ao.collision))

            for ao in attack_opportunities:
                if (
                        (ao.departure.action and not ao.departure.solo_capturer)
                        or target_fleet.attacked
                        or ao.departure.point.distance_from(ao.target_point) != ao.target_time
                ):
                    continue

                # if not ao.collision and patience_collision_opportunity:
                #     continue

                must_pass = [(ao.target_time, ao.target_point)]
                if ao.absorbed_ships > 0:
                    must_pass += ao.must_pass

                mining_sy = mine(agent, shipyards=[ao.departure], roundtrip_attack_mode=True, must_pass=must_pass,
                                 roundtrip_attack_target=target_fleet, absorbed_ships=ao.absorbed_ships,
                                 ignore_risk=ao.partial, last_attempt=ao.departure.last_attempt_attack,
                                 rescue=ao.rescue)
                if mining_sy:
                    msg = f"Step {board.step} {action_type} with mining: {ao.num_ships_to_launch} ships " \
                          f"{ao.departure.point}->{target_fleet.route.points()[ao.target_time - 1]}" \
                          f", distance={ao.target_time}"
                    if ao.absorbed_ships > 0:
                        msg += f', absorbing {ao.absorbed_ships}'
                    if ao.partial:
                        msg += f', partial interception!'
                    print(msg)
                    target_fleet.attacked = True
                    if ao.departure.solo_capturer:
                        print(f'step {board.step} overriding {ao.departure} capture decision to attack')
                        ao.departure.solo_capturer = False
                    break
                # else:
                #     print(f'{board.step} {ao.departure} -> {ao.target_point, ao.target_time} '
                #           f'no route for roundtrip attack. partial={ao.partial}')


    # Spawn for later attack
    for target_fleet in targets:
        # print(target_fleet,   len(immediate_attack_opportunities[target_fleet])
        #     ,len(latent_attack_opportunities[target_fleet])
        #     ,len(patience_attack_opportunities[target_fleet]))

        if (
            not target_fleet.attacked
            # len(immediate_attack_opportunities[target_fleet]) == 0
            and len(latent_attack_opportunities[target_fleet]) == 0
            and len(patience_attack_opportunities[target_fleet]) > 0
        ):
            ao = patience_attack_opportunities[target_fleet][0]
            sy = ao.departure
            if not sy.action:
                num_to_spawn = min(sy.max_ships_to_spawn, int(agent.available_kore() // board.spawn_cost))
                # if num_to_spawn > 0:
                msg = f'Step {board.step} {sy.point} spawning {num_to_spawn} for later roundtrip attack ' \
                      f'{target_fleet.ship_count} {target_fleet}@{target_fleet.point} at {ao.target_point}@{ao.target_time}, patience={ao.patience}'
                if ao.absorbed_ships > 0:
                    msg += f', absorbing {ao.absorbed_ships}'

                print(msg)
                sy.action = Spawn(num_to_spawn)


def is_intercept_direct_attack_route(route: BoardRoute, player: Player, direct_attack_fleet: Fleet,
                                     num_ships: int = None, absorb_ally: bool = False):
    board = player.board

    fleets = [FleetPointer(f) for f in board.fleets if f != direct_attack_fleet]

    for time, point in enumerate(route.points()[:-1]):
        for fleet in fleets:
            route_health = fleet.obj.route_health
            fleet.update()

            if fleet.point is None:
                continue

            if fleet.point == point:
                if not absorb_ally or fleet.obj.player_id != player.game_id:
                    return True
                else:
                    if time < len(route_health) and num_ships <= route_health[time]['health']:
                        return True

            # TODO instead of avoiding dmg completely, see if the fleet is able to sustain the dmg
            if fleet.obj.player_id != player.game_id:
                for p in fleet.point.adjacent_points:
                    if p == point:
                        return True

    return False


# @profile
def suicide_attack(agent: Player, max_distance: int = 10):
    board = agent.board

    max_distance = min(board.steps_left, max_distance)

    targets = _find_adjacent_targets(agent, max_distance)
    if not targets:
        return

    shipyards = [
        x for x in agent.shipyards if x.ship_count > 0 and not (x.action and not x.reinforcement_mode)
    ]

    if not shipyards:
        return

    fleets_to_be_attacked = set()

    incoming_attacking_fleets = []
    for sy in agent.shipyards:
        incoming_attacking_fleets += [f for f in sy.incoming_hostile_fleets]

    for t in sorted(targets, key=lambda x: (-len(x["fleets"]), x["time"])):
        target_point = t["point"]
        target_time = t["time"]
        target_fleets = t["fleets"]

        if any(x in fleets_to_be_attacked for x in target_fleets):
            continue

        for sy in shipyards:

            # if sy.action:
            #     continue

            if sy.reinforcement_mode:
                if sum([f in sy.reinforcement_target.incoming_hostile_fleets for f in target_fleets]) < 1:
                    continue

            if sy.defend_mode:
                if sum([f in sy.incoming_hostile_fleets for f in target_fleets]) < 1:
                    continue

            if sy.reinforcement_mode and sum([f in incoming_attacking_fleets for f in target_fleets]) < 1:
                continue

            distance = sy.distance_from(target_point)
            if distance > target_time:
                continue

            min_ship_count = min(x.ship_count for x in target_fleets)

            if len([f for f in target_fleets if f in sy.incoming_hostile_fleets]) > 0:
                # target fleets intersect with hostile fleets, use all ships
                num_ships_to_send = min(sy.ship_count, min_ship_count)
            else:
                num_ships_to_send = min(sy.available_ships, min_ship_count)

            routes = find_suicide_route(
                agent=agent,
                departure=sy.point,
                num_ships=num_ships_to_send,
                target_point=target_point,
                route_len=target_time
            )

            if not routes:
                continue

            # Improved: choose the route that has the least expected kore
            if len(routes) >= 2:
                routes.sort(key=lambda r: r.expected_kore_sparse(agent, num_ships_to_send))

            for route in routes:
                power_opti, power_pessi = agent.net_power((6, sy))
                for target_fleet in t['fleets']:
                    pad_right = max(0, 22 - 1 - target_fleet.expected_dmg_positions_np.shape[0])
                    compensate = np.concatenate([np.zeros(shape=(1, board.size, board.size)),
                                                 target_fleet.expected_dmg_positions_np,
                                                 np.zeros(shape=(pad_right, board.size, board.size))])[:22]
                    power_pessi += compensate

                if not is_protected_route(board,
                                          route, power_pessi,
                                          num_ships_to_send,
                                          horizon=target_time):

                    continue

                msg = f"Step {board.step} Suicide attack {sy.point}->{target_point}, " \
                      f"distance={distance}, target_time={target_time}, " \
                      f"fleet_count={num_ships_to_send}"
                if sy.reinforcement_mode:
                    msg += f'. Overriding reinforcement to suicide attack'
                print(msg)
                logger.info(msg)
                sy.action = Launch(num_ships_to_send, route)
                for fleet in target_fleets:
                    fleets_to_be_attacked.add(fleet)
                    fleet.attacked = True
                break

            if not sy.action:
                print(f'step {board.step}, {sy} {sy.point}->{target_point} no safe route for suicide attack, skip')


# @profile
def _find_adjacent_targets(agent: Player, max_distance: int = 5., min_targets=2):
    board = agent.board
    shipyards_points = {x.point for x in board.shipyards}
    fleets = [FleetPointer(f) for f in board.fleets]
    if len(fleets) < 2:
        return []

    time = 0
    targets = []
    while any(x.is_active for x in fleets) and time <= max_distance:
        time += 1

        for f in fleets:
            f.update()
            # if f.obj.suicide_attacker:
            #     targets.append({"point": f.point, "time": time, "fleets": [f.obj]})

        point_to_fleet = {
            x.point: x.obj
            for x in fleets
            if x.is_active and x.point not in shipyards_points and x.obj.player_id != agent.game_id
        }

        for point in board:
            if point in point_to_fleet or point in shipyards_points:
                continue

            adjacent_fleets = [point_to_fleet[x] for x in point.adjacent_points
                               if x in point_to_fleet
                               and len(point_to_fleet[x].route_health) >= time
                               and point_to_fleet[x].route_health[time - 1]['health'] > 2]

            # adjacent_enemy_fleets = [fleet for fleet in adjacent_fleets if fleet.player_id != agent.game_id]
            if len(adjacent_fleets) < min_targets:
                continue

            targets.append({"point": point, "time": time, "fleets": adjacent_fleets})

    return targets


def _need_more_ships(agent: Player, ship_count: int):
    board = agent.board
    if board.steps_left < 10:
        return False
    if ship_count > _max_ships_to_control(agent):
        return False
    if board.steps_left < 50 and is_invitable_victory(agent):
        return False
    return True


def _max_ships_to_control(agent: Player):
    return max(100, 3 * sum(x.ship_count for x in agent.opponents))


# @profile
def greedy_spawn(agent: Player):
    board = agent.board

    if board.step == 5:
        return

    for sy in sorted(agent.shipyards, key=lambda x: x.defend_mode, reverse=True):
        if not sy.action and sy.defend_mode and agent.available_kore() > sy.max_ships_to_spawn * board.configuration.spawn_cost:

            if sy.evacuate and sy.ship_count > 0:
                continue
            sy.action = Spawn(sy.max_ships_to_spawn)

    max_defense = 0
    for sy in board.shipyards:
        if sy.player_id != agent.game_id:
            max_defense = max(max_defense, max(sy.ship_count + sy.estimated_ship_counts((4, True, False, True))[1:4]))

    if not _need_more_ships(agent, agent.ship_count):
        return

    ship_count = agent.ship_count
    opponent_ship_count = sum(x.ship_count for x in agent.opponents)
    opponent_fleets = [f for f in board.fleets if f.player_id != agent.game_id]

    max_ship_count = _max_ships_to_control(agent)
    for shipyard in sorted(agent.shipyards, key=lambda sy: sy.frontier_risk, reverse=True):
        if shipyard.action:
            continue

        if shipyard.evacuate and shipyard.ship_count > 0:
            continue

        if max_defense < shipyard.available_ships and shipyard.available_ships >= 21:
            continue

        if shipyard.ship_count > agent.ship_count * 0.25 / len(agent.shipyards) and \
                ship_count > opponent_ship_count:  # TODO
            continue

        if shipyard.ship_count > (agent.ship_count + agent.kore // board.spawn_cost) * 0.4:
            continue

        if shipyard.ship_count > agent.ship_count / len(agent.shipyards) \
                or shipyard.ship_count > shipyard.max_ships_to_spawn * 10:
            continue

        if len(agent.fleets) == 0 and len(opponent_fleets) > 0:
            continue

        num_ships_to_spawn = shipyard.max_ships_to_spawn
        if int(agent.available_kore() // board.spawn_cost) >= num_ships_to_spawn:
            shipyard.action = Spawn(num_ships_to_spawn)

        ship_count += num_ships_to_spawn

        if ship_count > max_ship_count:
            return


# @profile
def spawn(agent: Player):
    board = agent.board

    if not _need_more_ships(agent, agent.ship_count):
        return

    ship_count = agent.ship_count
    max_ship_count = _max_ships_to_control(agent)
    for shipyard in sorted(agent.shipyards, key=lambda sy: (sy.defend_mode, sy.frontier_risk), reverse=True):
        if shipyard.action:
            continue
        num_ships_to_spawn = min(
            int(agent.available_kore() // board.spawn_cost),
            shipyard.max_ships_to_spawn,
        )
        if num_ships_to_spawn:
            shipyard.action = Spawn(num_ships_to_spawn)
            ship_count += num_ships_to_spawn
            if ship_count > max_ship_count:
                return
