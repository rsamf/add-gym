import torch
from torch import Tensor
import add_gym.util.torch_util as torch_util

class ADDReward:
    def __init__(self, config, env, add_obs, device):
        self.env = env
        self.add_obs = add_obs
        self.device = device
        self.config = config

        self._reward_pose_w = config.get("reward_pose_w")
        self._reward_vel_w = config.get("reward_vel_w")
        self._reward_root_pose_w = config.get("reward_root_pose_w")
        self._reward_root_vel_w = config.get("reward_root_vel_w")
        self._reward_key_pos_w = config.get("reward_key_pos_w")
        self._reward_pose_scale = config.get("reward_pose_scale")
        self._reward_vel_scale = config.get("reward_vel_scale")
        self._reward_root_pose_scale = config.get("reward_root_pose_scale")
        self._reward_root_vel_scale = config.get("reward_root_vel_scale")
        self._reward_key_pos_scale = config.get("reward_key_pos_scale")
        
        self._root_height_obs = config.get("root_height_obs", False)

        joint_err_w = config.get("joint_err_w", None)
        self._parse_joint_err_weights(joint_err_w)

    def _parse_joint_err_weights(self, joint_err_w):
        num_joints = self.env.robot._kin_char_model.get_num_joints()

        if joint_err_w is None:
            self._joint_err_w = torch.ones(
                num_joints - 1, device=self.device, dtype=torch.float32
            )
        else:
            self._joint_err_w = torch.tensor(
                joint_err_w, device=self.device, dtype=torch.float32
            )

        assert self._joint_err_w.shape[-1] == num_joints - 1

        dof_size = self.env.robot._kin_char_model.get_dof_size()
        self._dof_err_w = torch.zeros(
            dof_size, device=self.device, dtype=torch.float32
        )

        for j in range(1, num_joints):
            dof_dim = self.env.robot._kin_char_model.get_joint_dof_dim(j)
            if dof_dim > 0:
                curr_w = self._joint_err_w[j - 1]
                dof_idx = self.env.robot._kin_char_model.get_joint_dof_idx(j)
                self._dof_err_w[dof_idx : dof_idx + dof_dim] = curr_w

    def compute_reward(self):
        root_pos = self.env.robot.base_pos
        root_rot = self.env.robot.base_quat
        root_vel = self.env.robot.base_lin_vel
        root_ang_vel = self.env.robot.base_ang_vel
        dof_pos = self.env.robot.dof_pos
        dof_vel = self.env.robot.dof_vel

        track_root_h = self._root_height_obs
        track_root = self.add_obs._track_global_root()

        return compute_reward(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            tar_root_pos=self.add_obs.ref_root_pos,
            tar_root_rot=self.add_obs.ref_root_rot,
            tar_root_vel=self.add_obs.ref_root_vel,
            tar_root_ang_vel=self.add_obs.ref_root_ang_vel,
            tar_dof_pos=self.add_obs.ref_dof_pos,
            tar_dof_vel=self.add_obs.ref_dof_vel,
            dof_err_w=self._dof_err_w,
            track_root_h=track_root_h,
            track_root=track_root,
            pose_w=self._reward_pose_w,
            vel_w=self._reward_vel_w,
            root_pose_w=self._reward_root_pose_w,
            root_vel_w=self._reward_root_vel_w,
            pose_scale=self._reward_pose_scale,
            vel_scale=self._reward_vel_scale,
            root_pose_scale=self._reward_root_pose_scale,
            root_vel_scale=self._reward_root_vel_scale,
        )

@torch.jit.script
def convert_to_local_root(root_rot, root_vel, root_ang_vel):
    # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]

    heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)

    local_root_rot = torch_util.quat_mul(heading_inv_rot, root_rot)
    local_root_vel = torch_util.quat_rotate(heading_inv_rot, root_vel)
    local_root_ang_vel = torch_util.quat_rotate(heading_inv_rot, root_ang_vel)

    return local_root_rot, local_root_vel, local_root_ang_vel

@torch.jit.script
def compute_reward(
    root_pos,
    root_rot,
    root_vel,
    root_ang_vel,
    dof_pos,
    dof_vel,
    tar_root_pos,
    tar_root_rot,
    tar_root_vel,
    tar_root_ang_vel,
    tar_dof_pos,
    tar_dof_vel,
    dof_err_w,
    track_root_h,
    track_root,
    pose_w,
    vel_w,
    root_pose_w,
    root_vel_w,
    pose_scale,
    vel_scale,
    root_pose_scale,
    root_vel_scale,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, float, float, float, float, float, float, float, float) -> Tensor
    
    # Pose reward using dof_pos
    pose_diff = tar_dof_pos - dof_pos
    pose_err = torch.sum(dof_err_w * pose_diff * pose_diff, dim=-1)

    vel_diff = tar_dof_vel - dof_vel
    vel_err = torch.sum(dof_err_w * vel_diff * vel_diff, dim=-1)

    root_pos_diff = tar_root_pos - root_pos

    if not track_root:
        root_pos_diff[..., 0:2] = 0

    if not track_root_h:
        root_pos_diff[..., 2] = 0

    root_pos_err = torch.sum(root_pos_diff * root_pos_diff, dim=-1)

    if not track_root:
        root_rot, root_vel, root_ang_vel = convert_to_local_root(
            root_rot, root_vel, root_ang_vel
        )
        tar_root_rot, tar_root_vel, tar_root_ang_vel = convert_to_local_root(
            tar_root_rot, tar_root_vel, tar_root_ang_vel
        )

    root_rot_err = torch_util.quat_diff_angle(root_rot, tar_root_rot)
    root_rot_err *= root_rot_err

    root_vel_diff = tar_root_vel - root_vel
    root_vel_err = torch.sum(root_vel_diff * root_vel_diff, dim=-1)

    root_ang_vel_diff = tar_root_ang_vel - root_ang_vel
    root_ang_vel_err = torch.sum(root_ang_vel_diff * root_ang_vel_diff, dim=-1)

    pose_r = torch.exp(-pose_scale * pose_err)
    vel_r = torch.exp(-vel_scale * vel_err)
    root_pose_r = torch.exp(-root_pose_scale * (root_pos_err + 0.1 * root_rot_err))
    root_vel_r = torch.exp(-root_vel_scale * (root_vel_err + 0.1 * root_ang_vel_err))

    r = (
        pose_w * pose_r
        + vel_w * vel_r
        + root_pose_w * root_pose_r
        + root_vel_w * root_vel_r
    )

    return r
