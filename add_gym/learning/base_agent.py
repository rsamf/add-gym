import abc
import enum
import numpy as np
import os
import time
import torch
import add_gym.learning.experience_buffer as experience_buffer
import add_gym.learning.mp_optimizer as mp_optimizer
import add_gym.learning.normalizer as normalizer
import add_gym.util.tb_logger as tb_logger
import add_gym.util.torch_util as torch_util
import add_gym.learning.distribution_gaussian_diag as distribution_gaussian_diag
from add_gym.util.logger import Logger


class DoneFlags(enum.Enum):
    NULL = 0
    FAIL = 1
    SUCC = 2
    TIME = 3


class AgentMode(enum.Enum):
    TRAIN = 0
    TEST = 1


class BaseAgent(torch.nn.Module):
    NAME = "base"

    def __init__(self, config, env, device, distributed=False):
        super().__init__()

        self._env = env
        self._device = device
        self._iter = 0
        self._sample_count = 0
        self._config = config
        self._distributed = distributed
        self._load_params(config)

        self._build_normalizers()
        self._build_model(config)
        self.to(self._device)

        # Prepare model for distributed training
        if self._distributed:
            # Wrap model in DistributedDataParallel
            # We assume the device is already set correctly via torch.cuda.set_device()
            self._model = torch.nn.parallel.DistributedDataParallel(
                self._model,
                device_ids=[self._device],
                output_device=self._device,
                find_unused_parameters=config.get("distributed", {})
                .get("strategy", {})
                .get("ddp_find_unused_parameters", False),
            )

        self._build_optimizer(config)

        # No special optimizer wrapping needed for native DDP usually,
        # unless using specific libraries like Apex or ZeroRedundancyOptimizer
        # Standard PyTorch DDP handles gradient synchronization automatically during backward()

        self._build_exp_buffer(config)
        self._build_return_tracker()

        self._mode = AgentMode.TRAIN
        self._curr_obs = None
        self._curr_info = None
        self._is_restored = False

    @property
    def model(self):
        if self._distributed:
            return self._model.module
        return self._model

    def train_model(self, out_model_file, int_output_dir, log_file):
        max_samples = self._config.get("max_samples", int(1e6))
        start_time = time.time()

        self._curr_obs, self._curr_info = self._reset_envs()
        self._logger = self._build_logger(log_file)
        self._init_train()

        test_info = None

        while self._sample_count < max_samples:
            output_iter = self._iter % self._iters_per_output == 0

            if output_iter:
                test_info = self.test_model(self._test_episodes)

            train_info = self._train_iter()

            self._sample_count = self._update_sample_count()

            if self._sample_count >= max_samples:
                output_iter = True
                test_info = self.test_model(self._test_episodes)

            env_diag_info = self._env.get_diagnostics()
            self._log_train_info(train_info, test_info, env_diag_info, start_time)
            self._logger.print_log()

            if output_iter:
                self._logger.write_log()
                self._output_train_model(self._iter, out_model_file, int_output_dir)

                self._train_return_tracker.reset()
                self._curr_obs, self._curr_info = self._reset_envs()

            self._iter += 1

    def test_model(self, num_episodes):
        self.eval()
        self.set_mode(AgentMode.TEST)

        num_eps_proc = int(num_episodes)

        with torch.no_grad():
            self._curr_obs, self._curr_info = self._reset_envs()
            test_info = self._rollout_test(num_eps_proc)

        return test_info

    def get_action_size(self):
        a_space = self._env.robot.get_action_space()
        if a_space.dtype is torch.float:
            a_size = np.prod(a_space.shape)
        elif a_space.dtype is torch.int:
            a_size = 1
        else:
            assert False, "Unsuppoted action space: {}".format(a_space)
        return a_size

    def set_mode(self, mode):
        assert (
            mode == AgentMode.TRAIN or mode == AgentMode.TEST
        ), "Unsupported agent mode: {}".format(mode)
        self._mode = mode
        self._env.set_mode(self._mode)

    def get_num_envs(self):
        return self._env.num_envs

    def save(self, out_file):
        checkpoint = {
            "model": self.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "iter": self._iter,
            "sample_count": self._sample_count,
        }
        torch.save(checkpoint, out_file)

    def load(self, in_file):
        checkpoint = torch.load(in_file, map_location=self._device)
        self._is_restored = True

        if "model" in checkpoint and "optimizer" in checkpoint:
            state_dict = checkpoint["model"]
            self._optimizer.load_state_dict(checkpoint["optimizer"])
            self._iter = checkpoint.get("iter", 0)
            self._sample_count = checkpoint.get("sample_count", 0)
        else:
            state_dict = checkpoint
            self._is_restored = False

        # Check for DDP prefix mismatch
        model_keys = list(self.state_dict().keys())
        ckpt_keys = list(state_dict.keys())

        # If model expects DDP keys (start with '_model.module.') but checkpoint doesn't have them
        model_has_module = any(k.startswith("_model.module.") for k in model_keys)
        ckpt_has_module = any(k.startswith("_model.module.") for k in ckpt_keys)

        if model_has_module and not ckpt_has_module:
            Logger.print(
                "Converting single-process checkpoint to DDP format (adding .module prefix)"
            )
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("_model."):
                    # _model.layer -> _model.module.layer
                    new_k = k.replace("_model.", "_model.module.")
                    new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict

        elif not model_has_module and ckpt_has_module:
            Logger.print(
                "Converting DDP checkpoint to single-process format (removing .module prefix)"
            )
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("_model.module."):
                    # _model.module.layer -> _model.layer
                    new_k = k.replace("_model.module.", "_model.")
                    new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict

        self.load_state_dict(state_dict)

        Logger.print(f"Loaded model parameters from {in_file}")

    def calc_num_params(self):
        params = self.parameters()
        num_params = sum(p.numel() for p in params if p.requires_grad)
        return num_params

    def _load_params(self, config):
        self._discount = config["discount"]
        self._iters_per_output = config["iters_per_output"]
        self._normalizer_samples = config.get("normalizer_samples", np.inf)
        self._test_episodes = config["test_episodes"]
        self._steps_per_iter = config["steps_per_iter"]

    @abc.abstractmethod
    def _build_model(self, config):
        pass

    def _build_normalizers(self):
        obs_shape = self._add_obs.get_obs_shape()
        self._obs_norm = normalizer.Normalizer(
            obs_shape, device=self._device, dtype=torch.float
        )
        self._a_norm = self._build_action_normalizer()

    def _build_action_normalizer(self):
        a_space = self._env.robot.get_action_space()
        if a_space.dtype is torch.float:
            a_mean = torch.tensor(
                0.5 * (a_space[:, 1] + a_space[:, 0]),
                device=self._device,
                dtype=a_space.dtype,
            )
            a_std = torch.tensor(
                0.5 * (a_space[:, 1] - a_space[:, 0]),
                device=self._device,
                dtype=a_space.dtype,
            )
            a_norm = normalizer.Normalizer(
                a_mean.shape[:1],
                device=self._device,
                init_mean=a_mean,
                init_std=a_std,
                dtype=a_space.dtype,
            )
        elif a_space.dtype is torch.int:
            a_mean = torch.tensor([0], device=self._device, dtype=a_space.dtype)
            a_std = torch.tensor([1], device=self._device, dtype=a_space.dtype)
            a_norm = normalizer.Normalizer(
                a_mean.shape,
                device=self._device,
                init_mean=a_mean,
                init_std=a_std,
                min_std=0,
                dtype=a_space.dtype,
            )
        else:
            assert False, "Unsuppoted action space: {}".format(a_space)
        return a_norm

    def _build_optimizer(self, config):
        opt_config = config["optimizer"]
        params = list(self.parameters())
        params = [p for p in params if p.requires_grad]
        self._optimizer = mp_optimizer.MPOptimizer(opt_config, params)

    def _build_exp_buffer(self, config):
        buffer_length = self._get_exp_buffer_length()
        batch_size = self.get_num_envs()
        self._exp_buffer = experience_buffer.ExperienceBuffer(
            buffer_length=buffer_length, batch_size=batch_size, device=self._device
        )

        obs_space = self._add_obs.get_obs_shape()
        obs_buffer = torch.zeros(
            [buffer_length, batch_size] + list(obs_space),
            device=self._device,
            dtype=torch.float,
        )
        self._exp_buffer.add_buffer("obs", obs_buffer)

        next_obs_buffer = torch.zeros_like(obs_buffer)
        self._exp_buffer.add_buffer("next_obs", next_obs_buffer)

        a_space = self._env.robot.get_action_space()
        a_shape = list(a_space.shape[:1])
        if a_shape == []:
            a_shape = [1]

        action_buffer = torch.zeros(
            [buffer_length, batch_size] + a_shape,
            device=self._device,
            dtype=a_space.dtype,
        )
        self._exp_buffer.add_buffer("action", action_buffer)

        reward_buffer = torch.zeros(
            [buffer_length, batch_size], device=self._device, dtype=torch.float
        )
        self._exp_buffer.add_buffer("reward", reward_buffer)

        done_buffer = torch.zeros(
            [buffer_length, batch_size], device=self._device, dtype=torch.int
        )
        self._exp_buffer.add_buffer("done", done_buffer)

    def _build_return_tracker(self):
        self._train_return_tracker = ReturnTracker(self.get_num_envs(), self._device)
        self._test_return_tracker = ReturnTracker(self.get_num_envs(), self._device)

    @abc.abstractmethod
    def _get_exp_buffer_length(self):
        return 0

    def _build_logger(self, log_file):
        # Standard logging for single-instance training or main process in distributed
        # In distributed mode, we only log from the main process (rank 0)
        if self._distributed and torch.distributed.get_rank() != 0:
            from add_gym.util.logger import Logger

            return Logger()  # Dummy logger for non-main processes

        # Standard logging for single-instance training
        log = tb_logger.TBLogger()
        log.set_step_key("Samples")
        log.configure_output_file(log_file)
        return log

    def _update_sample_count(self):
        sample_count = self._exp_buffer.get_total_samples()
        return sample_count

    def _init_train(self):
        if not self._is_restored:
            self._iter = 0
            self._sample_count = 0
        else:
            Logger.print(
                f"Resuming training from iter {self._iter} sample {self._sample_count}"
            )

        self._exp_buffer.clear()
        self._train_return_tracker.reset()
        self._test_return_tracker.reset()

    def _train_iter(self):
        self._init_iter()

        self.eval()
        self.set_mode(AgentMode.TRAIN)

        with torch.no_grad():
            self._rollout_train(self._steps_per_iter)

        data_info = self._build_train_data()
        train_info = self._update_model()

        if self._need_normalizer_update():
            self._update_normalizers()

        info = {**train_info, **data_info}

        info["mean_return"] = self._train_return_tracker.get_mean_return().item()
        info["mean_ep_len"] = self._train_return_tracker.get_mean_ep_len().item()
        info["num_eps"] = self._train_return_tracker.get_episodes()

        return info

    def _init_iter(self):
        pass

    def _rollout_train(self, num_steps):
        for i in range(num_steps):
            action, action_info = self._decide_action(self._curr_obs, self._curr_info)
            self._record_data_pre_step(
                self._curr_obs, self._curr_info, action, action_info
            )

            next_obs, r, done, next_info = self._step_env(action)
            self._train_return_tracker.update(r, done)
            self._record_data_post_step(next_obs, r, done, next_info)

            self._curr_obs, self._curr_info = self._reset_done_envs(done)
            self._exp_buffer.inc()

    def _rollout_test(self, num_episodes):
        self._test_return_tracker.reset()

        if num_episodes == 0:
            test_info = {"mean_return": 0.0, "mean_ep_len": 0.0, "num_eps": 0}
        else:
            num_envs = self.get_num_envs()
            # minimum number of episodes to collect per env
            # this is mitigate bias in the return estimate towards shorter episodes
            min_eps_per_env = int(np.ceil(num_episodes / num_envs))

            while True:
                action, action_info = self._decide_action(
                    self._curr_obs, self._curr_info
                )

                next_obs, r, done, next_info = self._step_env(action)
                self._test_return_tracker.update(r, done)

                self._curr_obs, self._curr_info = self._reset_done_envs(done)

                eps_per_env = self._test_return_tracker.get_eps_per_env()
                if torch.all(eps_per_env > min_eps_per_env - 1):
                    break

            test_return = self._test_return_tracker.get_mean_return()
            test_ep_len = self._test_return_tracker.get_mean_ep_len()
            test_info = {
                "mean_return": test_return.item(),
                "mean_ep_len": test_ep_len.item(),
                "num_eps": self._test_return_tracker.get_episodes(),
            }
        return test_info

    @abc.abstractmethod
    def _decide_action(self, obs, info):
        a = None
        a_info = dict()
        return a, a_info

    def _step_env(self, action):
        obs, r, done, info = self._env.step(action)
        return obs, r, done, info

    def _record_data_pre_step(self, obs, info, action, action_info):
        self._exp_buffer.record("obs", obs)
        self._exp_buffer.record("action", action)

        if self._need_normalizer_update():
            self._obs_norm.record(obs)

    def _record_data_post_step(self, next_obs, r, done, next_info):
        self._exp_buffer.record("next_obs", next_obs)
        self._exp_buffer.record("reward", r)
        self._exp_buffer.record("done", done)

    def _reset_done_envs(self, done):
        done_indices = (done != DoneFlags.NULL.value).nonzero(as_tuple=False)
        env_ids = torch.flatten(done_indices)
        obs, info = self._reset_envs(env_ids)
        return obs, info

    def _reset_envs(self, env_ids=None):
        obs, info = self._env.reset(env_ids)
        return obs, info

    def _need_normalizer_update(self):
        return self._sample_count < self._normalizer_samples

    def _update_normalizers(self):
        self._obs_norm.update()

    def _build_train_data(self):
        return dict()

    @abc.abstractmethod
    def _update_model(self):
        pass

    def _compute_succ_val(self):
        r_succ = self._env.get_reward_succ()
        val_succ = r_succ / (1.0 - self._discount)
        return val_succ

    def _compute_fail_val(self):
        r_fail = self._env.get_reward_fail()
        val_fail = r_fail / (1.0 - self._discount)
        return val_fail

    def _log_train_info(self, train_info, test_info, env_diag_info, start_time):
        wall_time = (time.time() - start_time) / (60 * 60)  # store time in hours

        test_return = test_info["mean_return"]
        test_ep_len = test_info["mean_ep_len"]
        test_eps = test_info["num_eps"]

        train_return = train_info.pop("mean_return")
        train_ep_len = train_info.pop("mean_ep_len")
        train_eps = train_info.pop("num_eps")

        # In distributed mode, this is only called on the main process (rank 0) due to _build_logger logic
        self._logger.log("Iteration", self._iter, collection="1_Info")
        self._logger.log("Wall_Time", wall_time, collection="1_Info")
        self._logger.log("Samples", self._sample_count, collection="1_Info")

        self._logger.log("Test_Return", test_return, collection="0_Main")
        self._logger.log(
            "Test_Episode_Length", test_ep_len, collection="0_Main", quiet=True
        )
        self._logger.log("Test_Episodes", test_eps, collection="1_Info", quiet=True)

        self._logger.log("Train_Return", train_return, collection="0_Main")
        self._logger.log(
            "Train_Episode_Length", train_ep_len, collection="0_Main", quiet=True
        )
        self._logger.log("Train_Episodes", train_eps, collection="1_Info", quiet=True)

        for k, v in train_info.items():
            val_name = k.title()
            if torch.is_tensor(v):
                v = v.item()
            self._logger.log(val_name, v)

        for k, v in env_diag_info.items():
            val_name = k.title()
            if torch.is_tensor(v):
                v = v.item()
            self._logger.log(val_name, v, collection="2_Env", quiet=True)

    def _compute_action_bound_loss(self, norm_a_dist):
        loss = None
        action_space = self._env.robot.get_action_space()
        if action_space.dtype is torch.float:
            a_low = action_space[0]
            a_high = action_space[1]
            valid_bounds = torch.all(
                torch.isfinite(a_low) & torch.isfinite(a_high)
            ).item()

            if valid_bounds:
                assert isinstance(
                    norm_a_dist, distribution_gaussian_diag.DistributionGaussianDiag
                )
                # assume that actions have been normalized between [-1, 1]
                bound_min = -1
                bound_max = 1
                violation_min = torch.clamp_max(norm_a_dist.mode - bound_min, 0.0)
                violation_max = torch.clamp_min(norm_a_dist.mode - bound_max, 0)
                violation = torch.sum(torch.square(violation_min), dim=-1) + torch.sum(
                    torch.square(violation_max), dim=-1
                )
                loss = violation

        return loss

    def _output_train_model(self, iter, out_model_file, int_output_dir):
        # Save model checkpoint
        # This is only called on main process in distributed mode due to logic in main.py
        # BUT we double check here to be safe, as main.py logic might be bypassed or flawed
        if self._distributed and torch.distributed.get_rank() != 0:
            return

        self.save(out_model_file)

        if int_output_dir != "":
            int_model_file = os.path.join(
                int_output_dir, "model_{:010d}.pt".format(iter)
            )
            self.save(int_model_file)


class ReturnTracker:
    def __init__(self, num_envs, device):
        self._episodes = 0
        self._mean_return = torch.zeros([1], device=device, dtype=torch.float32)
        self._mean_ep_len = torch.zeros([1], device=device, dtype=torch.float32)

        self._return_buf = torch.zeros([num_envs], device=device, dtype=torch.float32)
        self._ep_len_buf = torch.zeros([num_envs], device=device, dtype=torch.long)
        self._eps_per_env_buf = torch.zeros([num_envs], device=device, dtype=torch.long)

    def get_mean_return(self):
        return self._mean_return

    def get_mean_ep_len(self):
        return self._mean_ep_len

    def get_episodes(self):
        return self._episodes

    def get_eps_per_env(self):
        return self._eps_per_env_buf

    def reset(self):
        self._episodes = 0
        self._eps_per_env_buf[:] = 0

        self._mean_return[:] = 0.0
        self._mean_ep_len[:] = 0.0

        self._return_buf[:] = 0.0
        self._ep_len_buf[:] = 0

    def update(self, reward, done):
        assert reward.shape == self._return_buf.shape
        assert done.shape == self._return_buf.shape

        self._return_buf += reward
        self._ep_len_buf += 1

        reset_mask = done != DoneFlags.NULL.value
        reset_ids = reset_mask.nonzero(as_tuple=False).flatten()
        num_resets = len(reset_ids)

        if num_resets > 0:
            new_mean_return = torch.mean(self._return_buf[reset_ids])
            new_mean_ep_len = torch.mean(self._ep_len_buf[reset_ids].type(torch.float))

            new_count = self._episodes + num_resets
            w_new = float(num_resets) / new_count
            w_old = float(self._episodes) / new_count

            self._mean_return = w_new * new_mean_return + w_old * self._mean_return
            self._mean_ep_len = w_new * new_mean_ep_len + w_old * self._mean_ep_len
            self._episodes += num_resets

            self._return_buf[reset_ids] = 0.0
            self._ep_len_buf[reset_ids] = 0
            self._eps_per_env_buf[reset_ids] += 1


def compute_td_lambda_return(r, next_vals, done, discount, td_lambda):
    assert r.shape == next_vals.shape

    return_t = torch.zeros_like(r)
    reset_mask = done != DoneFlags.NULL.value
    reset_mask = reset_mask.type(torch.float)

    last_val = r[-1] + discount * next_vals[-1]
    return_t[-1] = last_val

    timesteps = r.shape[0]
    for i in reversed(range(0, timesteps - 1)):
        curr_r = r[i]
        curr_reset = reset_mask[i]
        next_v = next_vals[i]
        next_ret = return_t[i + 1]

        curr_lambda = td_lambda * (1.0 - curr_reset)
        curr_val = curr_r + discount * (
            (1.0 - curr_lambda) * next_v + curr_lambda * next_ret
        )
        return_t[i] = curr_val

    return return_t
