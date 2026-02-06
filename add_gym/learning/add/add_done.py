import torch
from add_gym.learning.base_agent import DoneFlags
from add_gym.anim import motion
from add_gym.learning.add.add_observation import ADDObservation
from add_gym.learning.add.add_motion import ADDMotion

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from add_gym.envs.env import Environment

class ADDDone:
    def __init__(self, config: dict, env: "Environment", add_obs: ADDObservation, add_motion: ADDMotion, ground_plane, device):
        self.env = env
        self.add_obs = add_obs
        self.add_motion = add_motion
        self.device = device
        self.config = config
        self.ground_plane = ground_plane

        self._max_episode_length = config.get("max_episode_length", self.add_motion.motion_lib.get_total_length())
        self._enable_early_termination = config["enable_early_termination"]
        self._termination_height = config["termination_height"]
        self._pose_termination = config.get("pose_termination", False)
        self._pose_termination_dist = config.get("pose_termination_dist", 1.0)

        contact_bodies = config.get("contact_bodies", [])
        print("contact_bodies:")
        self._contact_body_ids = self._build_body_ids_tensor(contact_bodies)
        self._noncontact_body_ids = self._build_noncontact_body_ids_tensor(contact_bodies)

        self.done_buf = torch.zeros(
            (self.env.num_envs,), device=device, dtype=torch.int32
        )

    def _build_noncontact_body_ids_tensor(self, body_names):
        full_links = [l.idx for l in self.env.robot.entity.links]
        ignored_links = self._contact_body_ids
        noncontact_body_ids = [
            link_id for link_id in full_links if link_id not in ignored_links
        ]
        noncontact_body_ids = torch.tensor(
            noncontact_body_ids, device=self.device, dtype=torch.long
        )
        return noncontact_body_ids

    def _build_body_ids_tensor(self, body_names):
        body_ids = []
        for body_name in body_names:
            # body_id = self.env.robot.entity.get_link(name=body_name).idx_local
            body_id = self.env.robot.entity.get_link(name=body_name).idx
            print(f"  {body_name} =-> {body_id}")
            assert body_id != -1
            body_ids.append(body_id)

        body_ids = torch.tensor(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def compute_done(self):
        motion_times = self.add_obs._get_motion_times()
        motion_len = self.add_motion.get_motion_length(self.add_obs._motion_ids)
        motion_loop_mode = self.add_motion.get_motion_loop_mode(
            self.add_obs._motion_ids
        )
        motion_len_term = motion_loop_mode != motion.LoopMode.WRAP.value

        track_root = self.add_obs._track_global_root()

        root_pos = self.env.robot.base_pos
        dof_pos = self.env.robot.dof_pos
        ground_contact_forces = self.env.robot.get_ground_contact_forces_v2(self.ground_plane, self._noncontact_body_ids)
    
        self.done_buf[:] = compute_done(
            done_buf=self.done_buf,
            time=self.env.time_buf,
            ep_len=self._max_episode_length,
            root_pos=root_pos,
            dof_pos=dof_pos,
            tar_root_pos=self.add_obs.ref_root_pos,
            tar_dof_pos=self.add_obs.ref_dof_pos,
            ground_contact_force=ground_contact_forces,
            pose_termination=self._pose_termination,
            pose_termination_dist=self._pose_termination_dist,
            enable_early_termination=self._enable_early_termination,
            motion_times=motion_times,
            motion_len=motion_len,
            motion_len_term=motion_len_term,
            track_root=track_root,
        )
        return self.done_buf
    
    def reset_idx(self, env_ids):
        self.done_buf[env_ids] = DoneFlags.NULL.value


@torch.jit.script
def compute_done(
    done_buf,
    time,
    ep_len,
    root_pos,
    dof_pos,
    tar_root_pos,
    tar_dof_pos,
    ground_contact_force,
    pose_termination,
    pose_termination_dist,
    enable_early_termination,
    motion_times,
    motion_len,
    motion_len_term,
    track_root,
):
    # type: (Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, bool, float, bool, Tensor, Tensor, Tensor, bool) -> Tensor
    done = torch.full_like(done_buf, DoneFlags.NULL.value)
    timeout = time >= ep_len
    done[timeout] = DoneFlags.TIME.value

    motion_end = motion_times >= motion_len
    motion_end = torch.logical_and(motion_end, motion_len_term)
    done[motion_end] = DoneFlags.SUCC.value

    if enable_early_termination:
        failed = torch.zeros(done.shape, device=done.device, dtype=torch.bool)

        if ground_contact_force.shape[0] > 0:
            failed = torch.logical_or(failed, ground_contact_force)

        if pose_termination:
            dof_diff = tar_dof_pos - dof_pos
            dof_err = torch.mean(dof_diff**2, dim=-1)
            pose_fail = dof_err > pose_termination_dist

            if track_root:
                root_diff = tar_root_pos - root_pos
                root_err = torch.sum(root_diff**2, dim=-1)
                root_fail = root_err > pose_termination_dist
                pose_fail = torch.logical_or(pose_fail, root_fail)

            failed = torch.logical_or(failed, pose_fail)

        # only fail after first timestep
        not_first_step = time > 0.0
        failed = torch.logical_and(failed, not_first_step)
        done[failed] = DoneFlags.FAIL.value

    return done
