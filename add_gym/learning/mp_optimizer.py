import torch


class MPOptimizer:
    CHECK_SYNC_STEPS = 1000

    def __init__(self, config, param_list):
        self._param_list = param_list
        self._grad_list = None
        self._grad_clip = float(config.get("grad_clip", 0.0))
        self._optimizer = self._build_optimizer(config, param_list)
        self._steps = 0

    def step(self, loss: torch.Tensor):
        self._optimizer.zero_grad()

        loss.backward()

        if self._enable_grad_clip():
            self._clip_grads(self._grad_clip)

        self._optimizer.step()
        self._steps += 1

    def get_steps(self):
        return self._steps

    def _build_optimizer(self, config, param_list):
        lr = float(config["learning_rate"])
        weight_decay = float(config.get("weight_decay", 0.0))

        optimizer_type = config["type"]
        if optimizer_type == "SGD":
            optimizer = torch.optim.SGD(
                param_list, lr, momentum=0.9, weight_decay=weight_decay
            )
        elif optimizer_type == "Adam":
            optimizer = torch.optim.AdamW(param_list, lr, weight_decay=weight_decay)
        else:
            assert False, "Unsupported optimizer type: " + optimizer_type
        return optimizer

    def _enable_grad_clip(self):
        return self._grad_clip > 0.0

    def _clip_grads(self, max_norm):
        torch.nn.utils.clip_grad_norm_(self._param_list, max_norm)

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict)
