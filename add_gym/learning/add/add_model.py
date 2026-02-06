import torch
import add_gym.learning.nets.net_builder as net_builder
import add_gym.learning.ppo_model as ppo_model
import add_gym.util.torch_util as torch_util


class ADDModel(ppo_model.PPOModel):
    def __init__(self, config, env, obs_shape, action_space, d_obs_shape):
        super().__init__(config, env, obs_shape, action_space)
        self._build_disc(config, d_obs_shape)

    def eval_disc(self, disc_obs):
        h = self._disc_layers(disc_obs)
        val = self._disc_logits(h)
        return val

    def get_disc_logit_weights(self):
        return torch.flatten(self._disc_logits.weight)

    def get_disc_weights(self):
        weights = []
        for m in self._disc_layers.modules():
            if hasattr(m, "weight"):
                weights.append(torch.flatten(m.weight))

        weights.append(torch.flatten(self._disc_logits.weight))
        return weights

    def _build_nets(self, config, obs_shape, action_space):
        super()._build_nets(config, obs_shape, action_space)

    def _build_disc(self, config, d_obs_shape):
        init_output_scale = 1.0
        net_name = config["disc_net"]

        input_dict = self._build_disc_input_dict(d_obs_shape)
        self._disc_layers, layers_info = net_builder.build_net(
            net_name, input_dict, activation=self._activation
        )

        layers_out_size = torch_util.calc_layers_out_size(self._disc_layers)
        self._disc_logits = torch.nn.Linear(layers_out_size, 1)
        torch.nn.init.uniform_(
            self._disc_logits.weight, -init_output_scale, init_output_scale
        )
        torch.nn.init.zeros_(self._disc_logits.bias)

    def _build_disc_input_dict(self, d_obs_shape):
        input_dict = {"disc_obs": d_obs_shape}
        return input_dict
