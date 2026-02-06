import torch
import add_gym.util.torch_util as torch_util
import add_gym.util.circular_buffer as circular_buffer
from typing import TYPE_CHECKING
from add_gym.learning.add.add_motion import ADDMotion

if TYPE_CHECKING:
    from add_gym.envs.env import Environment


class ADDObservation:
    def __init__(self, config: dict, env: "Environment", motion: ADDMotion, device: str):
        self.env = env
        self.device = device
        self.config = config
        self.dt = env.ctrl_dt
        self.motion = motion

        self._enable_phase_obs = config.get("enable_phase_obs", True)
        self._enable_tar_obs = config.get("enable_tar_obs", False)
        self._enable_vel_obs = config.get("enable_vel_obs", False)
        self._tar_obs_steps = config.get("tar_obs_steps", [1])
        self._tar_obs_steps = torch.tensor(
            self._tar_obs_steps, device=device, dtype=torch.int
        )
        self._visualize_ref_char = config.get("visualize_ref_char", False)
        self._ref_char_offset = torch.tensor(
            config["ref_char_offset"], device=device, dtype=torch.float
        )
        self._global_obs = config.get("global_obs", False)
        self._root_height_obs = config.get("root_height_obs", False)
        self._num_disc_obs_steps = config["num_disc_obs_steps"]
        self._num_phase_encoding = config.get("num_phase_encoding", 0)

        self._build_sim_tensors(config, device)
        self._build_disc_obs_buffers()

        # Info dictionary to store obs for agent
        self.info = {}
        self.info["disc_obs"] = self._disc_obs_buf
        self.info["disc_obs_demo"] = self._disc_obs_demo_buf

    def _build_sim_tensors(self, config, device):
        num_envs = self.env.num_envs
        self._motion_ids = torch.zeros(num_envs, device=device, dtype=torch.int64)
        self._motion_time_offsets = torch.zeros(
            num_envs, device=device, dtype=torch.float32
        )
        root_pos = self.env.robot.base_pos
        root_rot = self.env.robot.base_quat
        root_vel = self.env.robot.base_lin_vel
        root_ang_vel = self.env.robot.base_ang_vel
        dof_pos = self.env.robot.dof_pos
        dof_vel = self.env.robot.dof_vel

        self.ref_root_pos = torch.zeros_like(root_pos)
        self.ref_root_rot = torch.zeros_like(root_rot)
        self.ref_root_vel = torch.zeros_like(root_vel)
        self.ref_root_ang_vel = torch.zeros_like(root_ang_vel)
        self.ref_dof_pos = torch.zeros_like(dof_pos)
        self.ref_dof_vel = torch.zeros_like(dof_vel)
        self.obs_buf = torch.zeros_like(self._compute_obs())

    def _build_body_ids_tensor(self, body_names):
        body_ids = []
        for body_name in body_names:
            body_id = self.env.robot.entity.get_link(name=body_name).idx_local
            print(f"  {body_name} =-> {body_id}")
            assert body_id != -1
            body_ids.append(body_id)

        body_ids = torch.tensor(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_disc_obs_buffers(self):
        num_envs = self.env.num_envs
        n = self._num_disc_obs_steps

        root_pos = self.env.robot.base_pos
        root_rot = self.env.robot.base_quat
        root_vel = self.env.robot.base_lin_vel
        root_ang_vel = self.env.robot.base_ang_vel
        dof_pos = self.env.robot.dof_pos
        dof_vel = self.env.robot.dof_vel

        self._disc_hist_root_pos = circular_buffer.CircularBuffer(
            batch_size=num_envs,
            buffer_len=n,
            shape=root_pos.shape[1:],
            dtype=root_pos.dtype,
            device=self.device,
        )

        self._disc_hist_root_rot = circular_buffer.CircularBuffer(
            batch_size=num_envs,
            buffer_len=n,
            shape=root_rot.shape[1:],
            dtype=root_rot.dtype,
            device=self.device,
        )

        self._disc_hist_root_vel = circular_buffer.CircularBuffer(
            batch_size=num_envs,
            buffer_len=n,
            shape=root_vel.shape[1:],
            dtype=root_vel.dtype,
            device=self.device,
        )

        self._disc_hist_root_ang_vel = circular_buffer.CircularBuffer(
            batch_size=num_envs,
            buffer_len=n,
            shape=root_ang_vel.shape[1:],
            dtype=root_ang_vel.dtype,
            device=self.device,
        )

        self._disc_hist_dof_pos = circular_buffer.CircularBuffer(
            batch_size=num_envs,
            buffer_len=n,
            shape=dof_pos.shape[1:],
            dtype=dof_pos.dtype,
            device=self.device,
        )

        self._disc_hist_dof_vel = circular_buffer.CircularBuffer(
            batch_size=num_envs,
            buffer_len=n,
            shape=dof_vel.shape[1:],
            dtype=dof_vel.dtype,
            device=self.device,
        )

        disc_obs_space = self.get_disc_obs_space()
        self._disc_obs_buf = torch.zeros(
            [num_envs] + list(disc_obs_space.shape[1:]),
            device=self.device,
            dtype=disc_obs_space.dtype,
        )
        self._disc_obs_demo_buf = torch.zeros_like(self._disc_obs_buf)

    def get_disc_obs_space(self):
        disc_obs = self.fetch_disc_obs_demo(self.env.num_envs)
        disc_obs_shape = list(disc_obs.shape[1:])
        disc_obs_space = torch.zeros((2, *disc_obs_shape), dtype=disc_obs.dtype)
        disc_obs_space[0, ...] = -torch.inf
        disc_obs_space[1, ...] = torch.inf
        return disc_obs_space

    def get_obs_shape(self):
        obs = self._compute_obs()
        return obs.shape[1:]

    def get_disc_obs_shape(self):
        disc_obs = self.fetch_disc_obs_demo(self.env.num_envs)
        return disc_obs.shape[1:]

    def fetch_disc_obs_demo(self, num_samples):
        motion_ids, motion_times0 = self.motion.sample_time(num_samples)
        disc_obs = self._compute_disc_obs_demo(motion_ids, motion_times0)
        return disc_obs

    def _update_ref_motion(self):
        motion_ids = self._motion_ids
        motion_times = self.env.time_buf + self._motion_time_offsets
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel = (
            self.motion.get_motion_step(motion_ids, motion_times)
        )
        self.ref_root_pos[:] = root_pos
        self.ref_root_rot[:] = root_rot
        self.ref_root_vel[:] = root_vel
        self.ref_root_ang_vel[:] = root_ang_vel
        self.ref_dof_vel[:] = dof_vel
        self.ref_dof_pos[:] = dof_pos
        if self._visualize_ref_char:
            qpos = torch.cat(
                [
                    self.ref_root_pos + self._ref_char_offset,
                    self.ref_root_rot,
                    self.ref_dof_pos,
                ],
                dim=-1,
            )
            self.env.robot.ref_entity.set_qpos(qpos)
            velocity = torch.cat(
                [
                    self.ref_root_vel,
                    self.ref_root_ang_vel,
                    self.ref_dof_vel,
                ],
                dim=-1,
            )
            self.env.robot.ref_entity.set_dofs_velocity(velocity)

    def _update_disc_hist(self):
        root_pos = self.env.robot.base_pos
        root_rot = self.env.robot.base_quat
        root_vel = self.env.robot.base_lin_vel
        root_ang_vel = self.env.robot.base_ang_vel
        dof_pos = self.env.robot.dof_pos
        dof_vel = self.env.robot.dof_vel
        self._disc_hist_root_pos.push(root_pos)
        self._disc_hist_root_rot.push(root_rot)
        self._disc_hist_root_vel.push(root_vel)
        self._disc_hist_root_ang_vel.push(root_ang_vel)
        self._disc_hist_dof_pos.push(dof_pos)
        self._disc_hist_dof_vel.push(dof_vel)

    def _fetch_tar_obs_data(self, motion_ids, motion_times):
        n = motion_ids.shape[0]
        num_steps = self._tar_obs_steps.shape[0]
        assert num_steps > 0

        motion_times = motion_times.unsqueeze(-1)
        motion_times = motion_times + self.dt * self._tar_obs_steps
        motion_ids_tiled = torch.broadcast_to(
            motion_ids.unsqueeze(-1), motion_times.shape
        )

        motion_ids_tiled = motion_ids_tiled.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, _, _, dof_pos, _ = (
            self.motion.get_motion_step(motion_ids_tiled, motion_times)
        )

        root_pos = root_pos.reshape([n, num_steps, root_pos.shape[-1]])
        root_rot = root_rot.reshape([n, num_steps, root_rot.shape[-1]])
        dof_pos = dof_pos.reshape([n, num_steps, dof_pos.shape[-1]])
        return root_pos, root_rot, dof_pos

    def _compute_obs(self):
        motion_ids = self._motion_ids
        motion_times = self._get_motion_times()

        root_pos = self.env.robot.base_pos
        root_rot = self.env.robot.base_quat
        root_vel = self.env.robot.base_lin_vel
        root_ang_vel = self.env.robot.base_ang_vel
        dof_pos = self.env.robot.dof_pos
        dof_vel = self.env.robot.dof_vel

        if self._enable_phase_obs:
            motion_phase = self.motion.get_motion_phase(motion_ids, motion_times)
        else:
            motion_phase = torch.zeros([0], device=self.device)

        if self._enable_tar_obs:
            tar_root_pos, tar_root_rot, tar_dof_pos = self._fetch_tar_obs_data(
                motion_ids, motion_times
            )
        else:
            tar_root_pos = torch.zeros([0], device=self.device)
            tar_root_rot = tar_root_pos
            tar_dof_pos = tar_root_pos

        obs = compute_add_obs(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            enable_vel_obs=self._enable_vel_obs,
            global_obs=self._global_obs,
            root_height_obs=self._root_height_obs,
            phase=motion_phase,
            num_phase_encoding=self._num_phase_encoding,
            enable_phase_obs=self._enable_phase_obs,
            enable_tar_obs=self._enable_tar_obs,
            tar_root_pos=tar_root_pos,
            tar_root_rot=tar_root_rot,
            tar_dof_pos=tar_dof_pos,
        )
        return obs

    def _update_disc_obs(self):
        root_pos = self._disc_hist_root_pos.get_all()
        root_rot = self._disc_hist_root_rot.get_all()
        root_vel = self._disc_hist_root_vel.get_all()
        root_ang_vel = self._disc_hist_root_ang_vel.get_all()
        dof_pos = self._disc_hist_dof_pos.get_all()
        dof_vel = self._disc_hist_dof_vel.get_all()

        disc_obs = compute_disc_obs(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            enable_vel_obs=self._enable_vel_obs,
            global_obs=self._global_obs,
        )
        self._disc_obs_buf[:] = disc_obs
        
    def update_motion(self):
        # update_misc
        self._update_ref_motion()
        self._update_disc_hist()

    def compute_obs(self):
        # update_observations
        self.obs_buf[:] = self._compute_obs()
        self._update_disc_obs()
        self._update_disc_obs_demo()
        return self.obs_buf

    def reset_idx(self, env_ids):
        n = len(env_ids)
        motion_ids, motion_times = self.motion.sample_time(n)
        self._motion_ids[env_ids] = motion_ids
        self._motion_time_offsets[env_ids] = motion_times
        self._update_ref_motion()
        qpos = torch.cat(
            [
                self.ref_root_pos[env_ids],
                self.ref_root_rot[env_ids],
                self.ref_dof_pos[env_ids],
            ],
            dim=-1,
        )
        self.env.robot.entity.set_qpos(qpos, envs_idx=env_ids)
        velocity = torch.cat(
            [
                self.ref_root_vel[env_ids],
                self.ref_root_ang_vel[env_ids],
                self.ref_dof_vel[env_ids],
            ],
            dim=-1,
        )
        self.env.robot.entity.set_dofs_velocity(velocity, envs_idx=env_ids)
        self._reset_disc_hist(env_ids)

    def _reset_disc_hist(self, env_ids):
        motion_ids = self._motion_ids[env_ids]
        motion_times = self.env.time_buf[env_ids] + self._motion_time_offsets[env_ids]
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel = self._fetch_disc_demo_data(motion_ids, motion_times)
        
        self._disc_hist_root_pos.fill(env_ids, root_pos)
        self._disc_hist_root_rot.fill(env_ids, root_rot)
        self._disc_hist_root_vel.fill(env_ids, root_vel)
        self._disc_hist_root_ang_vel.fill(env_ids, root_ang_vel)
        self._disc_hist_dof_pos.fill(env_ids, dof_pos)
        self._disc_hist_dof_vel.fill(env_ids, dof_vel)

    def get_observations(self):
        return self.obs_buf

    def _track_global_root(self):
        return self._enable_tar_obs and self._global_obs

    def _get_motion_times(self):
        motion_times = self.env.time_buf + self._motion_time_offsets
        return motion_times

    def _update_disc_obs_demo(self):
        motion_ids = self._motion_ids
        motion_times0 = self._get_motion_times()
        disc_obs = self._compute_disc_obs_demo(motion_ids, motion_times0)
        self._disc_obs_demo_buf[:] = disc_obs

    def _fetch_disc_demo_data(self, motion_ids, motion_times0):
        num_samples = motion_ids.shape[0]
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_disc_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -self.dt * torch.arange(
            0, self._num_disc_obs_steps, device=self.device
        )
        time_steps = torch.flip(time_steps, dims=[0])
        motion_times = motion_times + time_steps
        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel = (
            self.motion.get_motion_step(motion_ids, motion_times)
        )

        root_pos = torch.reshape(
            root_pos, shape=[num_samples, self._num_disc_obs_steps, root_pos.shape[-1]]
        )
        root_rot = torch.reshape(
            root_rot, shape=[num_samples, self._num_disc_obs_steps, root_rot.shape[-1]]
        )
        root_vel = torch.reshape(
            root_vel, shape=[num_samples, self._num_disc_obs_steps, root_vel.shape[-1]]
        )
        root_ang_vel = torch.reshape(
            root_ang_vel,
            shape=[num_samples, self._num_disc_obs_steps, root_ang_vel.shape[-1]],
        )
        dof_pos = torch.reshape(
            dof_pos,
            shape=[
                num_samples,
                self._num_disc_obs_steps,
                dof_pos.shape[-1],
            ],
        )
        dof_vel = torch.reshape(
            dof_vel, shape=[num_samples, self._num_disc_obs_steps, dof_vel.shape[-1]]
        )

        return root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel

    def _compute_disc_obs_demo(self, motion_ids, motion_times0):
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel = (
            self._fetch_disc_demo_data(motion_ids, motion_times0)
        )

        disc_obs = compute_disc_obs(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            enable_vel_obs=self._enable_vel_obs,
            global_obs=self._global_obs,
        )
        return disc_obs


@torch.jit.script
def compute_char_obs(
    root_pos,
    root_rot,
    root_vel,
    root_ang_vel,
    dof_pos,
    dof_vel,
    enable_vel_obs,
    global_obs,
    root_height_obs,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, bool) -> Tensor
    heading_rot = torch_util.calc_heading_quat_inv(root_rot)

    if global_obs:
        root_rot_obs = torch_util.quat_to_tan_norm(root_rot)
    else:
        local_root_rot = torch_util.quat_mul(heading_rot, root_rot)
        root_rot_obs = torch_util.quat_to_tan_norm(local_root_rot)

    obs = [root_rot_obs, dof_pos]

    if enable_vel_obs:
        if global_obs:
            root_vel_obs = root_vel
            root_ang_vel_obs = root_ang_vel
        else:
            root_vel_obs = torch_util.quat_rotate(heading_rot, root_vel)
            root_ang_vel_obs = torch_util.quat_rotate(heading_rot, root_ang_vel)
        obs += [root_vel_obs, root_ang_vel_obs, dof_vel]

    if root_height_obs:
        root_h = root_pos[:, 2:3]
        obs = [root_h] + obs

    obs = torch.cat(obs, dim=-1)
    return obs


@torch.jit.script
def compute_pos_obs(root_pos, root_rot, dof_pos, global_obs):
    # type: (Tensor, Tensor, Tensor, bool) -> Tensor

    root_pos_obs = root_pos.detach().clone()

    if not global_obs:
        root_pos_obs[..., 0:2] = 0.0

    root_rot_flat = torch.reshape(
        root_rot, [root_rot.shape[0] * root_rot.shape[1], root_rot.shape[2]]
    )
    root_rot_obs_flat = torch_util.quat_to_tan_norm(root_rot_flat)
    root_rot_obs = torch.reshape(
        root_rot_obs_flat,
        [root_rot.shape[0], root_rot.shape[1], root_rot_obs_flat.shape[-1]],
    )

    dof_pos_flat = torch.reshape(
        dof_pos,
        [
            dof_pos.shape[0] * dof_pos.shape[1],
            dof_pos.shape[2],
        ],
    )
    dof_pos_obs_flat = dof_pos_flat
    dof_pos_obs = torch.reshape(
        dof_pos_obs_flat,
        [
            dof_pos.shape[0],
            dof_pos.shape[1],
            dof_pos_obs_flat.shape[-1],
        ],
    )
    obs = [root_pos_obs, root_rot_obs, dof_pos_obs]
    obs = torch.cat(obs, dim=-1)

    return obs


@torch.jit.script
def compute_vel_obs(root_rot, root_vel, root_ang_vel, dof_vel, global_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor

    if not global_obs:
        heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)
        root_vel_obs = torch_util.quat_rotate(heading_inv_rot, root_vel)
        root_ang_vel_obs = torch_util.quat_rotate(heading_inv_rot, root_ang_vel)
    else:
        root_vel_obs = root_vel
        root_ang_vel_obs = root_ang_vel

    obs = [root_vel_obs, root_ang_vel_obs, dof_vel]
    obs = torch.cat(obs, dim=-1)

    return obs


@torch.jit.script
def compute_disc_obs(
    root_pos,
    root_rot,
    root_vel,
    root_ang_vel,
    dof_pos,
    dof_vel,
    enable_vel_obs,
    global_obs,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor

    pos_obs = compute_pos_obs(
        root_pos=root_pos,
        root_rot=root_rot,
        dof_pos=dof_pos,
        global_obs=global_obs,
    )

    disc_obs = pos_obs

    if enable_vel_obs:
        vel_obs = compute_vel_obs(
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel,
            global_obs=global_obs,
        )
        disc_obs = torch.cat([pos_obs, vel_obs], dim=-1)

    disc_obs = torch.reshape(disc_obs, [disc_obs.shape[0], -1])

    return disc_obs


@torch.jit.script
def compute_phase_obs(phase, num_phase_encoding):
    # type: (Tensor, int) -> Tensor
    phase_obs = phase.unsqueeze(-1)

    # positional embedding of phase
    if num_phase_encoding > 0:
        pe_exp = torch.arange(
            num_phase_encoding, device=phase.device, dtype=phase.dtype
        )
        pe_scale = 2.0 * torch.pi * torch.pow(2.0, pe_exp)
        pe_scale = pe_scale.unsqueeze(0)
        pe_val = phase.unsqueeze(-1) * pe_scale
        pe_sin = torch.sin(pe_val)
        pe_cos = torch.cos(pe_val)

        phase_obs = torch.cat((phase_obs, pe_sin, pe_cos), dim=-1)

    return phase_obs


@torch.jit.script
def compute_tar_obs(
    ref_root_pos,
    ref_root_rot,
    root_pos,
    root_rot,
    dof_pos,
    global_obs,
    root_height_obs,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    ref_root_pos = ref_root_pos.unsqueeze(-2)
    root_pos_obs = root_pos - ref_root_pos

    if not global_obs:
        heading_inv_rot = torch_util.calc_heading_quat_inv(ref_root_rot)
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand = heading_inv_rot_expand.repeat(
            (1, root_pos.shape[1], 1)
        )
        heading_inv_rot_flat = heading_inv_rot_expand.reshape(
            (
                heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1],
                heading_inv_rot_expand.shape[2],
            )
        )
        root_pos_obs_flat = torch.reshape(
            root_pos_obs,
            [root_pos_obs.shape[0] * root_pos_obs.shape[1], root_pos_obs.shape[2]],
        )
        root_pos_obs_flat = torch_util.quat_rotate(
            heading_inv_rot_flat, root_pos_obs_flat
        )
        root_pos_obs = torch.reshape(root_pos_obs_flat, root_pos.shape)

        root_rot = torch_util.quat_mul(heading_inv_rot_expand, root_rot)

    if root_height_obs:
        root_pos_obs[..., 2] = root_pos[..., 2]
    else:
        root_pos_obs = root_pos_obs[..., :2]

    root_rot_flat = torch.reshape(
        root_rot, [root_rot.shape[0] * root_rot.shape[1], root_rot.shape[2]]
    )
    root_rot_obs_flat = torch_util.quat_to_tan_norm(root_rot_flat)
    root_rot_obs = torch.reshape(
        root_rot_obs_flat,
        [root_rot.shape[0], root_rot.shape[1], root_rot_obs_flat.shape[-1]],
    )

    # dof pos
    dof_pos_flat = torch.reshape(
        dof_pos,
        [
            dof_pos.shape[0] * dof_pos.shape[1],
            dof_pos.shape[2],
        ],
    )
    dof_pos_obs_flat = dof_pos_flat
    dof_pos_obs = torch.reshape(
        dof_pos_obs_flat,
        [
            dof_pos.shape[0],
            dof_pos.shape[1],
            dof_pos_obs_flat.shape[-1],
        ],
    )

    obs = [root_pos_obs, root_rot_obs, dof_pos_obs]
    obs = torch.cat(obs, dim=-1)

    return obs


@torch.jit.script
def compute_add_obs(
    root_pos,
    root_rot,
    root_vel,
    root_ang_vel,
    dof_pos,
    dof_vel,
    enable_vel_obs,
    global_obs,
    root_height_obs,
    phase,
    num_phase_encoding,
    enable_phase_obs,
    enable_tar_obs,
    tar_root_pos,
    tar_root_rot,
    tar_dof_pos,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, bool, Tensor, int, bool, bool, Tensor, Tensor, Tensor) -> Tensor
    char_obs = compute_char_obs(
        root_pos=root_pos,
        root_rot=root_rot,
        root_vel=root_vel,
        root_ang_vel=root_ang_vel,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        enable_vel_obs=enable_vel_obs,
        global_obs=global_obs,
        root_height_obs=root_height_obs,
    )
    obs = [char_obs]

    if enable_phase_obs:
        phase_obs = compute_phase_obs(
            phase=phase, num_phase_encoding=num_phase_encoding
        )
        obs.append(phase_obs)

    if enable_tar_obs:
        if global_obs:
            ref_root_pos = root_pos
            ref_root_rot = root_rot
        else:
            ref_root_pos = tar_root_pos[..., 0, :]
            ref_root_rot = tar_root_rot[..., 0, :]

        tar_obs = compute_tar_obs(
            ref_root_pos=ref_root_pos,
            ref_root_rot=ref_root_rot,
            root_pos=tar_root_pos,
            root_rot=tar_root_rot,
            dof_pos=tar_dof_pos,
            global_obs=global_obs,
            root_height_obs=root_height_obs,
        )

        tar_obs = torch.reshape(
            tar_obs, [tar_obs.shape[0], tar_obs.shape[1] * tar_obs.shape[2]]
        )
        obs.append(tar_obs)

    obs = torch.cat(obs, dim=-1)

    return obs
