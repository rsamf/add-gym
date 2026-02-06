import torch
import add_gym.learning.amp_agent as amp_agent
import add_gym.learning.ppo_agent as ppo_agent
import add_gym.learning.diff_normalizer as diff_normalizer
from add_gym.learning.add.add_observation import ADDObservation
from add_gym.learning.add.add_reward import ADDReward
from add_gym.learning.add.add_done import ADDDone
from add_gym.learning.add.add_model import ADDModel
from add_gym.learning.add.add_motion import ADDMotion
from add_gym.envs.env import ImitationEnvironment


class ADDAgent(amp_agent.AMPAgent):
    NAME = "ADD"

    def __init__(self, env_config, distributed=False):
        if torch.cuda.is_available():
            # Device Masking active: Logical device is always 0
            device = "cuda:0"
        else:
            device = "cpu"
        print(f"Using device: {device}")
        self._env = ImitationEnvironment(env_config, device)
        self._add_motion = ADDMotion(env_config["task"], self._env, device)
        self._add_obs = ADDObservation(
            env_config["task"], self._env, self._add_motion, device
        )
        self._add_reward = ADDReward(
            env_config["task"], self._env, self._add_obs, device
        )
        self._add_done = ADDDone(
            env_config["task"],
            self._env,
            self._add_obs,
            self._add_motion,
            self._env.plane,
            device,
        )
        super().__init__(
            env_config["agent"], self._env, device, distributed=distributed
        )

        self._pos_diff = self._build_pos_diff()

    def _build_model(self, config):
        model_config = config["model"]
        self._model = ADDModel(
            model_config,
            self._env,
            self._add_obs.get_obs_shape(),
            self._env.robot.get_action_space(),
            self._add_obs.get_disc_obs_shape(),
        )

    def _build_pos_diff(self):
        disc_obs_space = self._add_obs.get_disc_obs_space()
        pos_diff = torch.zeros(
            disc_obs_space.shape[1], device=self._device, dtype=disc_obs_space.dtype
        )
        return pos_diff

    def _build_exp_buffer(self, config):
        super()._build_exp_buffer(config)

        buffer_length = self._get_exp_buffer_length()
        batch_size = self.get_num_envs()

        motion_id_buffer = torch.zeros(
            [buffer_length, batch_size], device=self._device, dtype=torch.long
        )
        self._exp_buffer.add_buffer("motion_ids", motion_id_buffer)

        motion_time_buffer = torch.zeros(
            [buffer_length, batch_size], device=self._device, dtype=torch.float
        )
        self._exp_buffer.add_buffer("motion_times", motion_time_buffer)

    def _build_normalizers(self):
        ppo_agent.PPOAgent._build_normalizers(self)

        disc_obs_space = self._add_obs.get_disc_obs_space()
        self._disc_obs_norm = diff_normalizer.DiffNormalizer(
            disc_obs_space.shape[1:], device=self._device, dtype=disc_obs_space.dtype
        )

    def _record_data_post_step(self, next_obs, r, done, next_info):
        ppo_agent.PPOAgent._record_data_post_step(self, next_obs, r, done, next_info)

        disc_obs = next_info["disc_obs"]
        disc_obs_demo = next_info["disc_obs_demo"]
        self._exp_buffer.record("disc_obs_demo", disc_obs_demo)
        self._exp_buffer.record("disc_obs", disc_obs)

        motion_ids = self._add_obs._motion_ids
        motion_times = self._add_obs._motion_time_offsets + self._env.time_buf
        self._exp_buffer.record("motion_ids", motion_ids)
        self._exp_buffer.record("motion_times", motion_times)

        if self._need_normalizer_update():
            obs_diff = disc_obs_demo - disc_obs
            self._disc_obs_norm.record(obs_diff)

    def _build_train_data(self):
        task_r = self._exp_buffer.get_data_flat("reward")

        disc_obs = self._exp_buffer.get_data_flat("disc_obs")
        disc_obs_demo = self._exp_buffer.get_data_flat("disc_obs_demo")
        disc_r = self._calc_disc_rewards(disc_obs=disc_obs, disc_obs_demo=disc_obs_demo)

        motion_ids = self._exp_buffer.get_data_flat("motion_ids")
        motion_times = self._exp_buffer.get_data_flat("motion_times")

        diff = disc_obs - disc_obs_demo
        diff_sq = torch.sum(torch.square(diff), dim=-1)
        self._add_motion.sampler.update_errors(motion_ids, motion_times, diff_sq)

        r = self._task_reward_weight * task_r + self._disc_reward_weight * disc_r
        self._exp_buffer.set_data_flat("reward", r)

        info = super(amp_agent.AMPAgent, self)._build_train_data()

        disc_reward_std, disc_reward_mean = torch.std_mean(disc_r)
        info["disc_reward_mean"] = disc_reward_mean
        info["disc_reward_std"] = disc_reward_std

        return info

    def _calc_disc_rewards(self, disc_obs, disc_obs_demo):
        obs_diff = disc_obs_demo - disc_obs
        norm_obs_diff = self._disc_obs_norm.normalize(obs_diff)
        reward = super()._calc_disc_rewards(norm_obs_diff)
        return reward

    def _compute_disc_loss(self, batch):
        disc_obs = batch["disc_obs"]
        tar_disc_obs = batch["disc_obs_demo"]

        pos_diff = self._pos_diff
        pos_diff = pos_diff.unsqueeze(dim=0)

        disc_pos_logit = self.model.eval_disc(pos_diff)
        disc_pos_logit = disc_pos_logit.squeeze(-1)

        diff_obs = tar_disc_obs - disc_obs
        norm_diff_obs = self._disc_obs_norm.normalize(diff_obs)
        norm_diff_obs.requires_grad_(True)
        disc_neg_logit = self.model.eval_disc(norm_diff_obs)
        disc_neg_logit = disc_neg_logit.squeeze(-1)

        disc_loss_pos = self._disc_loss_pos(disc_pos_logit)
        disc_loss_neg = self._disc_loss_neg(disc_neg_logit)
        disc_loss = 0.5 * (disc_loss_pos + disc_loss_neg)

        # logit reg
        logit_weights = self.model.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # grad penalty
        disc_neg_grad = torch.autograd.grad(
            disc_neg_logit,
            norm_diff_obs,
            grad_outputs=torch.ones_like(disc_neg_logit),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        disc_neg_grad = disc_neg_grad[0]
        grad_norm = torch.sqrt(torch.sum(torch.square(disc_neg_grad), dim=-1) + 1e-8)
        disc_grad_penalty = torch.mean(torch.square(grad_norm - 1))
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        # weight decay
        if self._disc_weight_decay != 0:
            disc_weights = self.model.get_disc_weights()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self._disc_weight_decay * disc_weight_decay

        disc_neg_acc, disc_pos_acc = self._compute_disc_acc(
            disc_neg_logit, disc_pos_logit
        )
        disc_pos_logit_mean = torch.mean(disc_pos_logit)
        disc_neg_logit_mean = torch.mean(disc_neg_logit)

        disc_info = {
            "disc_loss": disc_loss,
            "disc_grad_penalty": disc_grad_penalty.detach(),
            "disc_logit_loss": disc_logit_loss.detach(),
            "disc_pos_acc": disc_pos_acc.detach(),
            "disc_neg_acc": disc_neg_acc.detach(),
            "disc_pos_logit": disc_pos_logit_mean.detach(),
            "disc_neg_logit": disc_neg_logit_mean.detach(),
        }
        return disc_info

    def _step_env(self, action):
        # Step physics
        self._env.step(action)

        # Update new observations with incremented time step
        self._add_obs.update_motion()
        self._add_obs.compute_obs()
        # Reward based on new observations
        r = self._add_reward.compute_reward()
        # Check if done based on observations and time step
        done = self._add_done.compute_done()

        obs = self._add_obs.obs_buf
        info = self._add_obs.info

        return obs, r, done, info

    def _reset_envs(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._device)

        if len(env_ids) > 0:
            # Reset time
            self._env.reset(env_ids)
            # Reset ADD components
            self._add_done.reset_idx(env_ids)
            self._add_obs.reset_idx(env_ids)
            self._add_obs.compute_obs()

        return self._add_obs.obs_buf, self._add_obs.info

    def _log_train_info(self, train_info, test_info, env_diag_info, start_time):
        super()._log_train_info(train_info, test_info, env_diag_info, start_time)
        if self._iter % self._iters_per_output == 0:
            bins = self._add_motion.sampler.num_segments
            self._logger._writer.add_histogram(
                "Sampler/Errors", self._add_motion.sampler.errors, self._iter, bins=bins
            )
            self._logger._writer.add_histogram(
                "Sampler/Probs", self._add_motion.sampler.get_probs().cpu(), self._iter, bins=bins
            )

