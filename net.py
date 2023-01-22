import gym
import torch as th
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=128)
        hidden_dim = 128
        extractors = {}

        total_concat_size = 0

        extractors['image_block1'] = nn.Sequential(
            nn.Conv2d(10, hidden_dim, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 256, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, hidden_dim, kernel_size=3, stride=3, padding=0),
            nn.ReLU(), )

        extractors['image_block2'] = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten())
        total_concat_size += hidden_dim * 4

        # Run through a simple MLP
        extractors['vector'] = nn.Sequential(nn.Linear(observation_space['vector'].shape[0], 128),
                                             nn.ReLU())
        total_concat_size += 128

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        block1 = th.cat([self.extractors['image_block1'](observations[key]) for key in ['image_me', 'image_opponent']],
                        dim=1)

        encoded_tensor_list += [self.extractors['image_block2'](block1)]
        encoded_tensor_list += [self.extractors['vector'](observations['vector'])]

        return th.cat(encoded_tensor_list, dim=1)

