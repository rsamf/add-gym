import torch
import add_gym.learning.normalizer as normalizer
import add_gym.learning.ppo_agent as ppo_agent
import add_gym.util.torch_util as torch_util


class AMPAgent(ppo_agent.PPOAgent):
    NAME = "AMP"

    def __init__(self, config, env, device, distributed=False):
        super().__init__(config, env, device, distributed=distributed)

    def _load_params(self, config):
        super()._load_params(config)

        self._disc_loss_weight = config["disc_loss_weight"]
        self._disc_logit_reg = config["disc_logit_reg"]
        self._disc_grad_penalty = config["disc_grad_penalty"]
        self._disc_weight_decay = config["disc_weight_decay"]
        self._disc_reward_scale = config["disc_reward_scale"]
        self._disc_eval_batch_size = int(config.get("disc_eval_batch_size", 0))

        self._task_reward_weight = config["task_reward_weight"]
        self._disc_reward_weight = config["disc_reward_weight"]

    def _build_exp_buffer(self, config):
        super()._build_exp_buffer(config)

        buffer_length = self._get_exp_buffer_length()
        batch_size = self.get_num_envs()
        disc_obs_shape = self._add_obs.get_disc_obs_shape()
        disc_obs_buffer = torch.zeros(
            [buffer_length, batch_size] + list(disc_obs_shape),
            device=self._device,
            dtype=torch.float,
        )
        self._exp_buffer.add_buffer("disc_obs", disc_obs_buffer)
        disc_obs_demo_buffer = torch.zeros(
            [buffer_length, batch_size] + list(disc_obs_shape),
            device=self._device,
            dtype=torch.float,
        )
        self._exp_buffer.add_buffer("disc_obs_demo", disc_obs_demo_buffer)

    def _build_normalizers(self):
        super()._build_normalizers()
        disc_obs_shape = self._add_obs.get_disc_obs_shape()
        self._disc_obs_norm = normalizer.Normalizer(
            disc_obs_shape, device=self._device, dtype=torch.float
        )

    def _record_data_post_step(self, next_obs, r, done, next_info):
        super()._record_data_post_step(next_obs, r, done, next_info)

        disc_obs = next_info["disc_obs"]
        self._exp_buffer.record("disc_obs", disc_obs)

        if self._need_normalizer_update():
            self._disc_obs_norm.record(disc_obs)

    def _update_normalizers(self):
        super()._update_normalizers()
        self._disc_obs_norm.update()

    def _build_train_data(self):
        self._record_disc_demo_data()
        reward_info = self._compute_rewards()

        info = super()._build_train_data()
        info = {**info, **reward_info}
        return info

    def _record_disc_demo_data(self):
        task_r = self._exp_buffer.get_data_flat("reward")
        n = task_r.shape[0]
        disc_obs_demo = self._env.fetch_disc_obs_demo(n)
        self._exp_buffer.set_data_flat("disc_obs_demo", disc_obs_demo)
        self._disc_obs_norm.record(disc_obs_demo)

    def _compute_rewards(self):
        task_r = self._exp_buffer.get_data_flat("reward")

        disc_obs = self._exp_buffer.get_data_flat("disc_obs")
        norm_disc_obs = self._disc_obs_norm.normalize(disc_obs)
        disc_r = self._calc_disc_rewards(norm_disc_obs)

        r = self._task_reward_weight * task_r + self._disc_reward_weight * disc_r
        self._exp_buffer.set_data_flat("reward", r)

        disc_reward_std, disc_reward_mean = torch.std_mean(disc_r)
        info = {
            "disc_reward_mean": disc_reward_mean,
            "disc_reward_std": disc_reward_std,
        }

        return info

    def _compute_loss(self, batch):
        batch["norm_disc_obs"] = self._disc_obs_norm.normalize(batch["disc_obs"])
        batch["norm_disc_obs_demo"] = self._disc_obs_norm.normalize(
            batch["disc_obs_demo"]
        )

        info = super()._compute_loss(batch)

        disc_info = self._compute_disc_loss(batch)
        disc_loss = disc_info["disc_loss"]

        loss = info["loss"]
        loss = loss + self._disc_loss_weight * disc_loss
        info["loss"] = loss
        info = {**info, **disc_info}

        return info

    def _compute_disc_loss(self, batch):
        norm_disc_obs = batch["norm_disc_obs"]
        norm_disc_obs_demo = batch["norm_disc_obs_demo"]
        norm_disc_obs_demo.requires_grad_(True)

        disc_agent_logit = self.model.eval_disc(norm_disc_obs)
        disc_demo_logit = self.model.eval_disc(norm_disc_obs_demo)
        disc_agent_logit = disc_agent_logit.squeeze(-1)
        disc_demo_logit = disc_demo_logit.squeeze(-1)

        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # grad penalty
        disc_demo_grad = torch.autograd.grad(
            disc_demo_logit,
            norm_disc_obs_demo,
            grad_outputs=torch.ones_like(disc_demo_logit),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(
            disc_agent_logit, disc_demo_logit
        )

        disc_agent_logit_mean = torch.mean(disc_agent_logit)
        disc_demo_logit_mean = torch.mean(disc_demo_logit)

        disc_info = {
            "disc_loss": disc_loss,
            "disc_grad_penalty": disc_grad_penalty.detach(),
            "disc_agent_acc": disc_agent_acc.detach(),
            "disc_demo_acc": disc_demo_acc.detach(),
            "disc_agent_logit": disc_agent_logit_mean.detach(),
            "disc_demo_logit": disc_demo_logit_mean.detach(),
        }

        # logit reg
        if self._disc_logit_reg != 0:
            logit_weights = self.model.get_disc_logit_weights()
            disc_logit_loss = torch.sum(torch.square(logit_weights))
            disc_loss += self._disc_logit_reg * disc_logit_loss
            disc_info["disc_logit_loss"] = disc_logit_loss.detach()

        # weight decay
        if self._disc_weight_decay != 0:
            disc_weights = self.model.get_disc_weights()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self._disc_weight_decay * disc_weight_decay
            disc_info["disc_weight_decay"] = disc_weight_decay.detach()

        return disc_info

    def _disc_loss_neg(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.ones_like(disc_logits) * .1)
        return loss

    def _disc_loss_pos(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.ones_like(disc_logits) * .9)
        return loss

    def _compute_disc_acc(self, disc_agent_logit, disc_demo_logit):
        agent_acc = disc_agent_logit < 0
        agent_acc = torch.mean(agent_acc.float())
        demo_acc = disc_demo_logit > 0
        demo_acc = torch.mean(demo_acc.float())
        return agent_acc, demo_acc

    def _calc_disc_rewards(self, norm_disc_obs):
        with torch.no_grad():
            disc_inputs = {"disc_obs": norm_disc_obs}
            disc_logits = torch_util.eval_minibatch(
                self.model.eval_disc, disc_inputs, self._disc_eval_batch_size
            )
            disc_logits = disc_logits.squeeze(-1)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(
                torch.maximum(1 - prob, torch.tensor(0.0001, device=self._device))
            )
            disc_r *= self._disc_reward_scale
        return disc_r
