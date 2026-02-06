import genesis as gs
import torch

gs.init()

scene = gs.Scene(show_viewer=True, sim_options=gs.options.SimOptions(gravity=(0, 0, 0)))

plane = scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(
    morph=gs.morphs.MJCF(file="../assets/g1_description/g1_29.xml"),
)
motion = [
    0.000098,
    -0.000161,
    0.799087,
    0.004109,
    0.002375,
    0.021641,
    0.999755,
    -0.004409,
    0.060532,
    0.166431,
    0.176276,
    -0.097473,
    -0.023962,
    0.040313,
    -0.119414,
    -0.385817,
    0.206674,
    -0.180823,
    0.039829,
    0.016061,
    -0.006092,
    0.001142,
    -0.068885,
    1.474532,
    -0.172469,
    1.220329,
    0.281551,
    0.229778,
    0.196130,
    0.063265,
    -1.454644,
    0.000167,
    1.374207,
    -0.017476,
    0.069432,
    -0.162874,
]
motion = torch.tensor(motion, device="cpu").unsqueeze(0)
root_pos = motion[:, 0:3]
root_quat = motion[:, [6, 3, 4, 5]]
joint_pos = motion[:, 7:]

motion_order = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
joint_idx = (
    torch.tensor(
        [[robot.get_joint(name=name).dofs_idx[0] for name in motion_order]],
        device="cpu",
    )
    - 6
)
print(joint_idx)
print(joint_pos.shape)

joint_pos = torch.zeros_like(joint_pos).scatter_(1, joint_idx, joint_pos)
cam_0 = scene.add_camera()
scene.build(n_envs=1)
robot.set_pos(root_pos)
robot.set_quat(root_quat)
robot.set_dofs_position(joint_pos, dofs_idx_local=torch.arange(6, 35, device="cpu"))
scene.step()

import IPython

IPython.embed()
