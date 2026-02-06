import torch

import add_gym.learning.base_model as base_model
import add_gym.learning.nets.net_builder as net_builder
import add_gym.util.torch_util as torch_util


class PPOModel(base_model.BaseModel):
    def __init__(self, config, env, obs_shape, action_space):
        super().__init__(config, env)
        self._build_nets(config, obs_shape, action_space)

    def eval_actor(self, obs):
        h = self._actor_layers(obs)
        a_dist = self._action_dist(h)
        return a_dist

    def eval_critic(self, obs):
        h = self._critic_layers(obs)
        val = self._critic_out(h)
        return val

    def get_actor_params(self):
        params = list(self._actor_layers.parameters()) + list(
            self._action_dist.parameters()
        )
        return params

    def get_critic_params(self):
        params = list(self._critic_layers.parameters()) + list(
            self._critic_out.parameters()
        )
        return params

    def _build_nets(self, config, obs_shape, action_space):
        self._build_actor(config, obs_shape, action_space)
        self._build_critic(config, obs_shape)

    def _build_actor(self, config, obs_shape, action_space):
        net_name = config["actor_net"]
        input_dict = self._build_actor_input_dict(obs_shape)
        self._actor_layers, layers_info = net_builder.build_net(
            net_name, input_dict, activation=self._activation
        )

        self._action_dist = self._build_action_distribution(
            config, action_space, self._actor_layers
        )

    def _build_critic(self, config, obs_shape):
        net_name = config["critic_net"]
        input_dict = self._build_critic_input_dict(obs_shape)
        self._critic_layers, layers_info = net_builder.build_net(
            net_name, input_dict, activation=self._activation
        )

        layers_out_size = torch_util.calc_layers_out_size(self._critic_layers)
        self._critic_out = torch.nn.Linear(layers_out_size, 1)
        torch.nn.init.zeros_(self._critic_out.bias)

    def _build_actor_input_dict(self, obs_shape):
        input_dict = {"obs": obs_shape}
        return input_dict

    def _build_critic_input_dict(self, obs_shape):
        input_dict = {"obs": obs_shape}
        return input_dict
