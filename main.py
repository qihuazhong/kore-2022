# <--->
import pickle
import time
import os

import psutil

from helpers import RouteCache, MiningEfficiencyTracker
from quickboard import QuickBoard, Player
from logger import logger, init_logger
from offence import capture_shipyards
from defence import defend_shipyards, joint_defend_fleets
from expansion import expand
from mining import mine
from control import spawn, greedy_spawn, suicide_attack, roundtrip_attack
# from line_profiler_pycharm import profile

# <--->

IS_KAGGLE = os.path.exists("/kaggle_simulations")
if IS_KAGGLE:
    mr_file_path = '/kaggle_simulations/agent/mining_routes_13.pkl'
    rr_file_path = '/kaggle_simulations/agent/return_routes.pkl'
else:
    mr_file_path = './mining_routes_13.pkl'
    rr_file_path = './return_routes.pkl'
with open(mr_file_path, 'rb') as f:
    mining_routes = pickle.load(f)
with open(rr_file_path, 'rb') as f:
    return_routes = pickle.load(f)

route_cache = RouteCache()
efficiency_tracker = MiningEfficiencyTracker()

# @profile
def rl_agent(obs, conf, rl_action=None, test=False):
    # freeze_support()
    if obs["step"] == 0:
        init_logger(logger, f'./logs/{time.time()}.log')

    board = QuickBoard(obs, conf)
    step = board.step
    my_id = obs["player"]
    remaining_time = obs["remainingOverageTime"]
    logger.info(f"<step_{step + 1}>, remaining_time={remaining_time:.1f}")

    try:
        a: Player = board.get_player(my_id)
        a.preloaded_mining_routes = mining_routes
        a.preloaded_return_routes = return_routes
        a.route_cache = route_cache
        a.efficiency_tracker = efficiency_tracker
        a.set_supply_depots()

    except KeyError:
        return {}

    if not a.opponents:
        return {}

    defend_shipyards(a)
    capture_shipyards(a)
    expand(a)

    joint_defend_fleets(a)
    suicide_attack(a)
    roundtrip_attack(a)

    greedy_spawn(a)
    mine(a, default_max_distance=None)
    spawn(a)

    # print(route_cache.mining_routes.cache_info())
    # print('RAM memory % used:', psutil.virtual_memory())
    return a.actions(test)
