import types
# from line_profiler_pycharm import profile
from kaggle_environments.envs.kore_fleets.helpers import Observation, Configuration, Board, Direction, ShipyardAction
from kaggle_environments import make
import numpy as np
import torch
from ppo.main import rl_agent
from ppo.quickboard import QuickBoard

configuration: Configuration = Configuration({'episodeSteps': 400,
                                              'actTimeout': 3,
                                              'runTimeout': 9600,
                                              'startingKore': 2750,
                                              'size': 21,
                                              'spawnCost': 10.0,
                                              'convertCost': 50,
                                              'regenRate': 0.02,
                                               'maxRegenCellKore': 500,
                                              'agentTimeout': 60})


def process_obs(observation: Observation, board: Board):
    """process the raw observation and turn it into a dict of fixed size list/array, so that it can be easily integrated
    with stable-baselines models.

    Args:
        observation: the raw observation
        board: basically the same information as the observation, but in a more human-readable format

    Returns:

    """
    size = board.configuration.size
    spawn_cost = configuration.spawn_cost
    convert_cost = configuration.convert_cost
    player_id = board.current_player_id

    obs = {}
    my_fleets_ship_count = [0] * size ** 2
    my_fleets_cargo = [0] * size ** 2
    my_fleet_direction = [[0] * size ** 2] * 4
    my_fleet_collection_rate = [0.0] * size ** 2
    my_shipyards_max_spawn = [0.0] * size ** 2
    my_shipyards_ship_count = [0.0] * size ** 2

    opponent_ship_count = [0] * size ** 2
    opponent_cargo = [0] * size ** 2
    opponent_fleet_direction = [[0] * size ** 2] * 4
    opponent_fleet_collection_rate = [0.0] * size ** 2
    opponent_shipyards_max_spawn = [0.0] * size ** 2
    opponent_shipyards_ship_count = [0.0] * size ** 2

    for key, fleet in board.fleets.items():
        if fleet.player_id == player_id:
            my_fleets_ship_count[fleet.position.to_index(configuration.size)] = fleet.ship_count * 10
            my_fleets_cargo[fleet.position.to_index(configuration.size)] = fleet.kore
            my_fleet_direction[fleet.direction.to_index()][fleet.position.to_index(configuration.size)] = 100
            my_fleet_collection_rate[fleet.position.to_index(configuration.size)] = fleet.collection_rate * 100

        else:
            opponent_ship_count[fleet.position.to_index(configuration.size)] = fleet.ship_count * 10
            opponent_cargo[fleet.position.to_index(configuration.size)] = fleet.kore
            opponent_fleet_direction[fleet.direction.to_index()][fleet.position.to_index(configuration.size)] = 100
            opponent_fleet_collection_rate[fleet.position.to_index(configuration.size)] = fleet.collection_rate * 100

    for key, shipyard in board.shipyards.items():
        if shipyard.player_id == board.current_player_id:
            my_shipyards_max_spawn[shipyard.position.to_index(configuration.size)] = shipyard.max_spawn * 10
            my_shipyards_ship_count[shipyard.position.to_index(configuration.size)] = shipyard.ship_count * 10
        else:
            opponent_shipyards_max_spawn[shipyard.position.to_index(configuration.size)] = shipyard.max_spawn * 10
            opponent_shipyards_ship_count[shipyard.position.to_index(configuration.size)] = shipyard.ship_count * 10

    obs['image_me'] = np.array([observation['kore'], my_fleets_ship_count, my_fleets_cargo, my_fleet_collection_rate,
                                my_shipyards_max_spawn, my_shipyards_ship_count] +
                               my_fleet_direction, dtype=np.float).reshape((10, configuration.size, configuration.size))

    obs['image_opponent'] = np.array(
        [observation['kore'], opponent_ship_count, opponent_cargo, opponent_fleet_collection_rate,
         opponent_shipyards_max_spawn, opponent_shipyards_ship_count]
        + opponent_fleet_direction, dtype=np.float).reshape((10, configuration.size, configuration.size))

    quickboard = QuickBoard(observation, board.configuration)
    vector_obs = []
    for pl in quickboard.players:
        cargo = sum(f.kore for f in pl.fleets)
        kore = pl.kore
        ships = sum(f.ship_count for f in pl.fleets) + sum(f.ship_count for f in pl.shipyards)
        sys = len([sy for sy in pl.shipyards if sy.expected_build_time <= 0])
        networth = pl.kore + cargo + ships * spawn_cost + sys * spawn_cost * convert_cost
        if pl.fleets:
            avg_route_len = sum(len(f.route) for f in pl.fleets) / len(pl.fleets)
        else:
            avg_route_len = 0
        vector_obs += [networth, kore, cargo, ships, sys, avg_route_len]

    if len(quickboard.players) < 2:
        if quickboard.players[0].game_id == 0:
            vector_obs = vector_obs + [0, 0, 0, 0, 0, 0]
        else:
            vector_obs = [0, 0, 0, 0, 0, 0] + vector_obs

    obs['vector'] = np.array(vector_obs)
    return obs


def interpret_action(action: np.ndarray, board: Board, obs: Observation) -> dict:
    rl_action = {'max_distance': action[0] + 6}

    return rl_agent(obs, configuration, rl_action)


def custom_reward(obs: Observation, last_obs: Observation, old_board: Board, new_board: Board):
    """

    Args:
        obs:
        last_obs:
        old_board:
        new_board:

    Returns:

    """

    config: Configuration = old_board.configuration
    player_id = old_board.current_player_id

    baseline = sum([opponent.kore for opponent in new_board.opponents]) - \
               sum([opponent.kore for opponent in old_board.opponents])

    my_fleet = (sum([fleet.ship_count for key, fleet in new_board.fleets.items() if fleet.player_id == player_id]) -
                sum([fleet.ship_count for key, fleet in old_board.fleets.items() if
                     fleet.player_id == player_id])) * config.spawn_cost
    opponent_fleet = (sum([fleet.ship_count for key, fleet in new_board.fleets.items() if
                           fleet.player_id != player_id]) -
                      sum([fleet.ship_count for key, fleet in old_board.fleets.items() if
                           fleet.player_id != player_id])) * config.spawn_cost

    my_cargo = sum([fleet.kore for key, fleet in new_board.fleets.items() if fleet.player_id == player_id]) - \
               sum([fleet.kore for key, fleet in old_board.fleets.items() if fleet.player_id == player_id])

    opponent_cargo = sum([fleet.kore for key, fleet in new_board.fleets.items() if fleet.player_id != player_id]) - \
                     sum([fleet.kore for key, fleet in old_board.fleets.items() if fleet.player_id != player_id])

    my_shipyards = (sum([shipyard.ship_count + config.convert_cost for key, shipyard in new_board.shipyards.items() if
                         shipyard.player_id == player_id]) -
                    sum([shipyard.ship_count + config.convert_cost for key, shipyard in old_board.shipyards.items() if
                         shipyard.player_id == player_id])) \
                   * config.spawn_cost

    opponent_shipyards = (sum([shipyard.ship_count + config.convert_cost for key, shipyard in
                               new_board.shipyards.items() if shipyard.player_id != player_id]) -
                          sum([shipyard.ship_count + config.convert_cost for key, shipyard in
                               old_board.shipyards.items() if shipyard.player_id != player_id])) \
                         * config.spawn_cost

    # print(f'baseline:{baseline}, my_fleet:{my_fleet},opponent_fleet:{opponent_fleet}, my_cargo:{my_cargo}, '
    #       f'opponent_cargo:{opponent_cargo}, my_shipyards:{my_shipyards}, opponent_shipyards:{opponent_shipyards}')

    return - baseline + (
            my_fleet - opponent_fleet + my_cargo - opponent_cargo + my_shipyards - opponent_shipyards) * 0.0


def kore_trainer_wrapper(trainer):
    trainer.wrappee_step = trainer.step

    def wrapped_step(self, action):
        old_board = Board(trainer.last_obs, configuration)
        action = interpret_action(action, old_board, trainer.last_obs)
        # print(action)
        obs, reward, done, info = self.wrappee_step(action)

        # common_obs = {common_key: observation[common_key] for common_key in ('players', 'step', 'kore')}
        new_board = Board(obs, configuration)
        # print(trainer.last_obs)

        reward += custom_reward(obs, trainer.last_obs, old_board=old_board, new_board=new_board)
        trainer.last_obs = obs

        obs = process_obs(obs, new_board)
        return [obs, reward, done, info]

    trainer.step = types.MethodType(wrapped_step, trainer)

    trainer.wrappee_reset = trainer.reset

    def wrapped_reset(self):
        obs = self.wrappee_reset()
        board = Board(obs, configuration)
        trainer.last_obs = obs
        return process_obs(obs, board)

    trainer.reset = types.MethodType(wrapped_reset, trainer)

    return trainer
