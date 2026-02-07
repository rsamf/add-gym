from typing import TYPE_CHECKING

import torch
from add_gym.anim import motion_lib
from add_gym.learning.sampler import AdaptiveSegmentSampler


if TYPE_CHECKING:
    from add_gym.envs.env import Environment


class ADDMotion:
    def __init__(self, config: dict, env: "Environment", device: str):
        self.env = env
        self.device = device
        self.config = config

        self.motion_lib = motion_lib.MotionLib(
            motion_file=config["motion_file"],
            motion_order=list(config["motion_joint_order"]),
            kin_char_model=env.robot._kin_char_model,
            dt=self.env.ctrl_dt,
            device=device,
        )
        sampler_config = config.get("sampler", {})
        num_disc_obs_steps = config.get("num_disc_obs_steps", 1)
        self.sampler = AdaptiveSegmentSampler(
            clip_lengths=self.motion_lib.get_motion_lengths(),
            dt=self.env.ctrl_dt,
            num_segments=sampler_config.get("num_segments", 20),
            temperature=sampler_config.get("temperature", None),
            min_start_time=(num_disc_obs_steps - 1) * self.env.ctrl_dt,
        )
        self._rand_reset = config.get("rand_reset", True)

    def get_motion_step(self, motion_ids, motion_times):
        return self.motion_lib.get_precomputed_motion_step(motion_ids, motion_times)

    def get_motion_phase(self, motion_ids, motion_times):
        return self.motion_lib.calc_motion_phase(motion_ids, motion_times)
    
    # For ADDDone
    def get_motion_length(self, motion_ids):
        return self.motion_lib.get_motion_length(motion_ids)
    
    # For ADDDone
    def get_motion_loop_mode(self, motion_ids):
        return self.motion_lib.get_motion_loop_mode(motion_ids)

    def sample_motions(self, n):
        return self.motion_lib.sample_motions(n)

    def sample_time(self, n):
        motion_ids = self.sample_motions(n)

        if self._rand_reset:
            motion_times = self.sampler.sample_start_frame(motion_ids)
        else:
            motion_times = torch.zeros(n, dtype=torch.float, device=self.device)

        return motion_ids, motion_times
    