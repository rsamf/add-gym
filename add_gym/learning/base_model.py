import torch

import add_gym.learning.distribution_gaussian_diag as distribution_gaussian_diag
import add_gym.learning.distribution_categorical as distribution_categorical
import add_gym.util.torch_util as torch_util


class BaseModel(torch.nn.Module):
    def __init__(self, config, env):
        super().__init__()
        self._activation = torch.nn.ReLU

    def _build_action_distribution(self, config: dict, a_space, input):
        in_size = torch_util.calc_layers_out_size(input)
        
        if a_space.dtype is torch.float:
            a_size = a_space.shape[0]
            a_init_output_scale = config["actor_init_output_scale"]
            a_std_type = distribution_gaussian_diag.StdType[config["actor_std_type"]]
            a_std = config["action_std"]
            a_dist = distribution_gaussian_diag.DistributionGaussianDiagBuilder(in_size, a_size, std_type=a_std_type,
                                                                            init_std=a_std, init_output_scale=a_init_output_scale)
        elif a_space.dtype is torch.int:
            num_actions = a_space.n
            a_init_output_scale = config["actor_init_output_scale"]
            a_dist = distribution_categorical.DistributionCategoricalBuilder(in_size, num_actions, 
                                                                             init_output_scale=a_init_output_scale)
        else:
            assert(False), "Unsuppoted action space: {}".format(a_space)

        return a_dist