import random
from typing import List

import numpy as np
from collections import defaultdict
# from line_profiler_pycharm import profile

# <--->
from basic import max_ships_to_spawn, cached_property, cached_call
from helpers import absorbables, sustained_dmg
from mining import find_suicide_route
from quickboard import Player, Shipyard, Launch, Spawn
from logger import logger


# <--->


class _ShipyardTarget:

    # @profile
    def __init__(self, shipyard: Shipyard):
        self.shipyard = shipyard
        self.point = shipyard.point
        self.attacked = False
        self._future_ship_count = self._estimate_future_ship_count()

    def __repr__(self):
        return f"Target {self.shipyard}"

    @cached_call
    def estimate_shipyard_power(self, time):
        return self._future_ship_count[time]

    def _get_total_incoming_power(self):
        return sum(x.ship_count for x in self.shipyard.incoming_allied_fleets)

    def _get_reinforcement_distance(self):
        incoming_allied_fleets = self.shipyard.incoming_allied_fleets
        if not incoming_allied_fleets:
            return np.inf
        return min(x.eta for x in incoming_allied_fleets)

    # def _estimate_profit(self):
    #     board = self.shipyard.board
    #     spawn_cost = board.spawn_cost
    #     profit = sum(
    #         2 * x.expected_kore() - x.ship_count * spawn_cost
    #         for x in self.shipyard.incoming_allied_fleets
    #     )
    #     profit += spawn_cost * board.shipyard_cost
    #     return profit

    # @profile
    def _estimate_future_ship_count(self) -> List[int]:
        """
        time -> ship count

        """
        shipyard = self.shipyard
        player = shipyard.player
        board = shipyard.board

        incoming_kore = np.sum([sy.incoming_kore for sy in player.shipyards], axis=0)

        # confirmed reinforcements
        confirmed_shipyard_reinforcements = defaultdict(int)
        for f in shipyard.incoming_allied_fleets:
            confirmed_shipyard_reinforcements[len(f.route)] += f.ship_count

        spawn_cost = board.spawn_cost
        player_kore = player.kore
        ship_count = shipyard.ship_count
        future_ship_count = [ship_count]
        for t in range(1, board.size + 1):
            ship_count += confirmed_shipyard_reinforcements[t]
            player_kore += incoming_kore[t]

            can_spawn = max_ships_to_spawn(shipyard.turns_controlled + t)
            spawn_count = min(int(player_kore // spawn_cost), can_spawn)
            player_kore -= spawn_count * spawn_cost
            ship_count += spawn_count
            future_ship_count.append(ship_count)
        return future_ship_count


def estimate_max_reinforcement_power(target: _ShipyardTarget,
                                     potential_reinforcement_shipyards: List[_ShipyardTarget],
                                     distance: int) -> int:
    max_reinforcement_power = 0
    for r_sy in potential_reinforcement_shipyards:
        # TODO: the estimation is not taking into account
        #  1): the opponent may earn more kore by launching more ships
        #  2): the opponent core is over-consumed by separate shipyards
        reinforcement_distance = target.shipyard.point.distance_from(r_sy.shipyard.point)
        max_wait_time_before_sending_reinforcement = max(0, distance - reinforcement_distance - 0)
        max_reinforcement_power += r_sy.estimate_shipyard_power(max_wait_time_before_sending_reinforcement)

    return max_reinforcement_power


# @profile
def capture_shipyards(agent: Player, max_attack_distance=10):
    board = agent.board
    agent_shipyards = [
        x for x in agent.shipyards if x.available_ships >= 3 and not x.action and not x.reinforcement_mode
    ]

    max_patience = 4

    if not agent_shipyards:
        return

    targets = []
    for op_sy in board.shipyards:
        if op_sy.player_id == agent.game_id:  # or op_sy.incoming_hostile_fleets:  # TODO
            continue

        target = _ShipyardTarget(op_sy)
        # if target.expected_profit > 0:
        targets.append(target)

    if len(targets) <= 3:
        max_attack_distance += 3

    if not targets:
        return

    for t in sorted(targets, key=lambda x: x.shipyard.frontier_proximity):

        if t.shipyard.incoming_hostile_fleets:
            eta = min(min(f.eta for f in t.shipyard.incoming_hostile_fleets), board.size)
            if sum(f.ship_count for f in t.shipyard.incoming_hostile_fleets if f.eta <= eta) > t.estimate_shipyard_power(eta):
                continue

        for distance in range(1, max_attack_distance+1):

            if t.attacked:
                continue

            shipyards_in_range_immediate = sorted([sy for sy in agent.shipyards
                                                   if sy.point.distance_from(t.point) <= distance and not sy.action],
                                                  key=lambda sy: sy.available_ships,
                                                  reverse=True)

            shipyards_in_range_latent = sorted([sy for sy in agent.shipyards
                                                   if sy.point.distance_from(t.point) <= distance],
                                                  key=lambda sy: sy.available_ships,
                                                  reverse=True)

            if not shipyards_in_range_immediate:
                continue

            for patience in range(max_patience):

                # conservative = (agent.ship_count < 0.8 * sum(op.ship_count for op in agent.opponents)) * 1
                #
                # if conservative and sum(
                #         f.ship_count for f in t.shipyard.incoming_allied_fleets if f.eta > distance+patience) <= 0:
                #     continue

                potential_reinforcement_shipyards = [sy_t for sy_t in targets
                                                     if 0 < sy_t.shipyard.point.distance_from(t.point) < distance+patience]

                power = t.estimate_shipyard_power(distance+patience)
                max_reinforcement_power = estimate_max_reinforcement_power(
                    t, potential_reinforcement_shipyards, distance+patience)
                late_reinforcement = sum(t.shipyard.incoming_allied_ships[distance+patience+1:])
                attack_en_route = sum(f.ship_count for f in t.shipyard.incoming_hostile_fleets if f.eta <= distance+patience)

                min_ships_to_takeover = power + max_reinforcement_power - attack_en_route


                if patience == 0:
                    incoming_ships = 0
                    if sum(sy.available_ships for sy in shipyards_in_range_immediate
                           ) + incoming_ships <= min_ships_to_takeover:
                        continue
                    # num_ships_to_launch = min(sy.available_ship_count, int(power * 1.2))  # TODO

                    joint_attack = {}
                    for idx, sy in enumerate(shipyards_in_range_immediate):
                        sy.must_pass = None
                        num_ships_to_launch = min(sy.available_ships,
                                                  int(min_ships_to_takeover*1.2 + 3 + late_reinforcement))  # TODO

                        max_current_step_absorbable, max_next_step_absorbable = 0, 0
                        absorbed_fleet = None
                        for f in [f for f in agent.fleets if not f.planned_absorbed]:
                            current_step_absorbable, next_step_absorbable, _, _ = absorbables(sy, f, t.point)
                            if current_step_absorbable and f.ship_count > max_current_step_absorbable:
                                max_current_step_absorbable = f.ship_count
                                sy.must_pass = current_step_absorbable
                                absorbed_fleet = f
                            if next_step_absorbable and f.ship_count > max_next_step_absorbable:
                                max_next_step_absorbable = f.ship_count

                        routes = find_suicide_route(
                            agent=agent,
                            departure=sy.point,
                            num_ships=num_ships_to_launch,
                            target_point=t.point,
                            route_len=distance,
                            max_sustained_dmg=max(0, sy.available_ships-min_ships_to_takeover)
                        )

                        if routes:
                            route_to_score = {}
                            for route in routes:
                                route_to_score[route] = route.expected_kore_sparse(agent, num_ships_to_launch)

                            routes = sorted(route_to_score, key=lambda r: -route_to_score[r])

                            absorbed_power = 0
                            chosen_route = None
                            # random_route = random.choice(routes)
                            if sy.must_pass and absorbed_fleet:
                                for route in routes:
                                    time, point = sy.must_pass[0]
                                    _, rows, cols = route.sparse_rep
                                    if time <= len(route) and (rows[time - 1] == point.x and cols[time - 1] == point.y):
                                        absorbed_power = max_current_step_absorbable
                                        absorbed_fleet.planned_absorbed = True
                                        chosen_route = route
                            if chosen_route is None:
                                chosen_route = routes[0]
                                # random_route

                            joint_attack[sy] = {'route': chosen_route, 'random_route': routes[0],
                                                'ships': num_ships_to_launch + sustained_dmg(agent, chosen_route),
                                                'absorbed_power': absorbed_power}

                    total_ship_power = sum(item['ships'] + item['absorbed_power'] for sy, item in joint_attack.items())

                    # print(t, total_ship_power, min_ships_to_takeover)
                    # print(joint_attack)
                    if total_ship_power >= min_ships_to_takeover:

                        for sy, item in joint_attack.items():
                            if sy.point.distance_from(t.point) > 9:
                                print(f'step {board.step}, long distance ({sy.point.distance_from(t.point)}) capture')

                            if min_ships_to_takeover >= 0:

                                if item['ships'] > min_ships_to_takeover or sy.must_pass is None:
                                    sy.action = Launch(item['ships'], item['random_route'])
                                    print(f'Step {board.step} \t{sy.point} --{item["ships"]}--> {t.point}')
                                else:
                                    sy.action = Launch(int(max(item['ships'], item["absorbed_power"]+1)), item['route'])
                                    sy.must_pass = None
                                    print(f'Step {board.step} \t{sy.point} --{item["ships"]}--> {t.point}, '
                                          f'absorbing {item["absorbed_power"]}')
                                if len(joint_attack.keys()) == 1:
                                    sy.solo_capturer = True
                                min_ships_to_takeover -= item['ships'] + item['absorbed_power']
                        t.attacked = True
                        break

                else:

                    spawning_plan = {}
                    for sy in shipyards_in_range_latent:
                        spawning_plan[sy] = {'incoming_ships': sum(f.ship_count for f in sy.incoming_allied_fleets if f.eta <= patience)}
                        # incoming_ships += sum(f.ship_count for f in sy.incoming_allied_fleets if f.eta <= patience)
                    # TODO consider incoming kore
                    num_ships_to_spawn = min(int((agent.available_kore() +
                                                  sum(agent.expected_kore_by_time[:patience+1])) // board.spawn_cost),
                                             sum(sy.max_ships_to_spawn for sy in shipyards_in_range_latent))

                    total_max_power = sum(sy.available_ships for sy in shipyards_in_range_immediate) \
                        + sum(item['incoming_ships'] for sy, item in spawning_plan.items()) + num_ships_to_spawn

                    if total_max_power < min_ships_to_takeover:
                        # No capture opportunity
                        continue

                    # if sum(sy.available_ships for sy in shipyards_in_range_latent
                    #        ) + sum(item['incoming_ships'] for sy, item in spawning_plan.items()) + num_ships_to_spawn < min_ships_to_takeover:
                    #     # No capture opportunity
                    #     continue

                    if sum(item['incoming_ships'] for sy, item in spawning_plan.items()) >= min_ships_to_takeover:
                        # Only incoming ships are enough, no need to spawn
                        continue

                    min_ships_to_takeover -= sum(item['incoming_ships'] for sy, item in spawning_plan.items())

                    for sy in shipyards_in_range_immediate:
                        if min_ships_to_takeover < 0:
                            continue

                        num_ships_to_spawn = min(sy.max_ships_to_spawn,
                                                 int(agent.available_kore() // board.spawn_cost))

                        if (
                            (num_ships_to_spawn > 0 and min_ships_to_takeover > 0)
                            or sy.point.distance_from(t.point) < distance
                        ):
                            msg = f"Step {board.step} Spawning {num_ships_to_spawn} for later attack " \
                                  f"{sy.point}->{t.point}, patience={patience}"
                            print(msg)
                            logger.info(msg)
                            sy.action = Spawn(num_ships_to_spawn)
                            min_ships_to_takeover -= num_ships_to_spawn + sy.available_ships
                    break
