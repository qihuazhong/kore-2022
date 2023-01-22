from kaggle_environments import make

from ppo.quickboard import QuickBoard

config = {'episodeSteps': 400,
          'actTimeout': 3,
          'runTimeout': 9600,
          'startingKore': 2750,
          'size': 21,
          'spawnCost': 10.0,
          'convertCost': 50,
          'regenRate': 0.02,
          'maxRegenCellKore': 500,
          'agentTimeout': 60,
          }


env = make("kore_fleets", debug=True, configuration=config)

trainer = env.train([None, 'do_nothing'])
# trainer = kore_trainer_wrapper(trainer)

total_rewards = []


obs = trainer.reset()
board = QuickBoard(obs, config)

print(board)