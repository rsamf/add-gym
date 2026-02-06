import torch
import re
import logging
from typing import List, Dict, Tuple
import add_gym.anim.kin_char_model as kin_char_model
import os.path
from add_gym.engine.base_engine import BaseEngine, BaseScene

logger = logging.getLogger(__name__)


class Manipulator:
    def __init__(
        self,
        num_envs: int,
        scene: BaseScene,
        engine: BaseEngine,
        robot_cfg: dict,
        env_spacing: float,
        enable_ref: bool = False,
        device: str = "cpu",
    ):
        # == set members ==
        self._device = device
        self._scene = scene
        self._engine = engine
        self._num_envs = num_envs
        self._enable_ref = enable_ref
        self._args = robot_cfg
        self._env_spacing = env_spacing

        # == Robot entity creation through engine abstraction ==
        morph_path = robot_cfg["urdf_path"]
        morph_type = "urdf" if morph_path.endswith(".urdf") else "mjcf"

        self._build_kin_char_model(robot_cfg["urdf_path"])

        # Create robot entity
        self._robot_entity = scene.add_entity(
            morph_type=morph_type,
            morph_file=morph_path,
            morph_pos=(0.0, 0.0, 0.0),
            morph_quat=(1.0, 0.0, 0.0, 0.0),
            material_type="rigid",
        )

        # Create reference entity if needed
        self._ref_entity = None
        if self._enable_ref:
            self._ref_entity = scene.add_entity(
                morph_type=morph_type,
                morph_file=morph_path,
                morph_pos=(0.0, 0.0, 0.0),
                morph_quat=(1.0, 0.0, 0.0, 0.0),
                material_type="rigid",
                visualize_contact=False,
            )

        # Build joint limits
        self.joint_limits = []
        for joint in self._robot_entity.joints:
            self.joint_limits.extend(joint.dofs_limit)
        self.joint_limits = torch.tensor(
            self.joint_limits, device=self._engine.device, dtype=self._engine.tc_float
        )  # (num_dofs, 2)

        self.gain_scale = robot_cfg.get("gain_scale", 1.0)
        self.link_cfg = robot_cfg["links"]
        self.joint_cfg = robot_cfg["joints"]
        self.link_lookup, self.joint_lookup = self._build_lookups()
        self.non_root_joints = [
            i
            for i in range(self._robot_entity.n_dofs)
            if i not in self.joint_lookup["base"]
        ]

    def _build_kin_char_model(self, char_file):
        _, file_ext = os.path.splitext(char_file)
        if file_ext == ".xml":
            char_model = kin_char_model.KinCharModel(self._device)
        else:
            print("Unsupported character file format: {:s}".format(file_ext))
            assert False

        self._kin_char_model = char_model
        self._kin_char_model.load_char_file(char_file)

    def _build_lookups(self) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        link_lookup: Dict[str, List[int]] = {}
        for link in self._robot_entity.links:
            is_link_tagged = False
            for cfg in self.link_cfg:
                pattern = cfg["match"]
                tags = cfg["tags"]
                if re.fullmatch(pattern, link.name):
                    for tag in tags:
                        if tag in link_lookup:
                            link_lookup[tag].append(link.idx_local)
                        else:
                            link_lookup[tag] = [link.idx_local]
                    is_link_tagged = True
            if not is_link_tagged:
                logger.warning(f"link {link.name} does not match any pattern in config")

        joint_lookup: Dict[str, List[int]] = {}
        for joint in self._robot_entity.joints:
            is_joint_tagged = False
            for cfg in self.joint_cfg:
                pattern = cfg["match"]
                tags = cfg["tags"]
                if re.fullmatch(pattern, joint.name):
                    for tag in tags:
                        if tag in joint_lookup:
                            joint_lookup[tag].extend(joint.dofs_idx)
                        else:
                            joint_lookup[tag] = [*joint.dofs_idx]
                    is_joint_tagged = True
            if not is_joint_tagged:
                logger.warning(
                    f"joint {joint.name} does not match any pattern in config"
                )

        return link_lookup, joint_lookup

    def _override_default_angles(self, angles: torch.Tensor) -> torch.Tensor:
        if "default_angles" in self._args:
            overrides = self._args["default_angles"].items()
            for joint_name, angle in overrides:
                dof_idx = self._robot_entity.get_joint(joint_name).dofs_idx_local
                angles[dof_idx] = angle
        return angles

    def _set_gains(self):
        gains = (
            torch.ones(self._robot_entity.n_dofs, device=self._engine.device, dtype=torch.float32)
            * 100.0
        )
        ankle_idx = self.joint_lookup.get("ankle")
        knee_idx = self.joint_lookup.get("knee")
        hip_idx = self.joint_lookup.get("hip")
        core_idx = self.joint_lookup.get("core")
        arm_idx = self.joint_lookup.get("arm")
        hand_idx = self.joint_lookup.get("hand", [])  # optional
        assert ankle_idx is not None, "No ankle joints found in config"
        assert knee_idx is not None, "No knee joints found in config"
        assert hip_idx is not None, "No hip joints found in config"
        assert core_idx is not None, "No core joints found in config"
        assert arm_idx is not None, "No arm joints found in config"
        check_en = [False] * self._robot_entity.n_dofs
        for i in [*ankle_idx, *knee_idx, *hip_idx, *core_idx, *arm_idx, *hand_idx]:
            check_en[i] = True
        if not all(check_en[6:]):
            raise ValueError("Some joints are not assigned gains in the config")
        gains[ankle_idx] = 120.0
        gains[knee_idx] = 120.0
        gains[hip_idx] = 80.0
        gains[core_idx] = 50.0
        gains[arm_idx] = 50.0
        gains[hand_idx] = 20.0
        gains *= self.gain_scale
        dampings = 2.0 * torch.sqrt(gains)
        self.entity.set_dofs_kp(gains)
        self.entity.set_dofs_kv(dampings)

    def on_build(self):
        min_z = (
            -torch.min(self._robot_entity.get_AABB()[0, :, 2])
            + 0.001
            + self._robot_entity.get_pos()[0][2]
        )
        self.base_init_pos = torch.tensor([0.0, 0.0, min_z], device=self._engine.device)
        self.base_init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._engine.device)

        self.default_joint_angles = self._override_default_angles(
            self.joint_positions[0]
        )
        self.non_root_default_joint_angles = self.default_joint_angles[
            self.non_root_joints
        ]
        self._set_gains()
        self.reset()

    def get_action_space(self) -> torch.Tensor:
        def _build_action_bounds_pos(dof_low: torch.Tensor, dof_high: torch.Tensor):
            low = torch.zeros(dof_high.shape, dtype=torch.float32)
            high = torch.zeros(dof_high.shape, dtype=torch.float32)

            num_joints = self._kin_char_model.get_num_joints()
            for j in range(1, num_joints):
                curr_joint = self._kin_char_model.get_joint(j)
                j_low = curr_joint.get_joint_dof(dof_low)
                j_high = curr_joint.get_joint_dof(dof_high)

                curr_mid = 0.5 * (j_high + j_low)

                diff_high = torch.abs(j_high - curr_mid)
                diff_low = torch.abs(j_low - curr_mid)
                curr_scale = torch.maximum(diff_high, diff_low)
                curr_scale *= 1.4

                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                curr_joint.set_joint_dof(curr_low, low)
                curr_joint.set_joint_dof(curr_high, high)

            return low, high

        limits = self.joint_limits[self.non_root_joints]
        low, high = _build_action_bounds_pos(limits[:, 0], limits[:, 1])
        ret = torch.stack([low, high], dim=1)
        return ret

    def get_ground_contact_forces(self) -> torch.Tensor:
        contact_forces = self._robot_entity.get_links_net_contact_force()
        links_height = self.link_pos[..., 2]
        above_ground = links_height > 0.15
        contact_forces[above_ground, :] = 0.0
        return contact_forces
    
    def get_ground_contact_forces_v2(self, surface_plane, contact_idx) -> torch.Tensor:
        contacts = self._robot_entity.get_contacts(with_entity=surface_plane, exclude_self_contact=True)
        links_a = contacts['link_a']
        links_b = contacts['link_b']
        valid_mask = contacts['valid_mask']
        contact_links_a = torch.isin(links_a, contact_idx)
        contact_links_a = torch.logical_and(contact_links_a, valid_mask)
        contact_links_b = torch.isin(links_b, contact_idx)
        contact_links_b = torch.logical_and(contact_links_b, valid_mask)
        
        return torch.logical_or(contact_links_a.any(dim=1), contact_links_b.any(dim=1))

    def joints_by_tag(self, tag: str) -> List[int]:
        return self.joint_lookup[tag]

    def links_by_tag(self, tag: str) -> List[int]:
        return self.link_lookup[tag]

    def reset(self, envs_idx: torch.IntTensor | None = None):
        # Get default joint angles
        if envs_idx is None:
            envs_idx = torch.arange(self._num_envs, device=self._engine.device)

        n = len(envs_idx)
        self._robot_entity.set_dofs_position(
            self.non_root_default_joint_angles.expand((n, -1)),
            envs_idx=envs_idx,
            dofs_idx_local=self.non_root_joints,
        )

        # set above ground
        pos = self.base_init_pos.expand((n, -1)).clone()
        self._robot_entity.set_pos(
            pos,
            envs_idx=envs_idx,
        )
        self._robot_entity.set_quat(
            self.base_init_quat.expand((n, -1)),
            envs_idx=envs_idx,
        )
        self._robot_entity.zero_all_dofs_velocity(envs_idx=envs_idx)

    def apply_action(self, action: torch.Tensor, allowed_action_idx=None) -> None:
        self._robot_entity.control_dofs_position(
            position=action,
            dofs_idx_local=(
                allowed_action_idx if allowed_action_idx else self.non_root_joints
            ),
        )

    @property
    def base_pos(self):
        return self._robot_entity.get_pos()

    @property
    def base_quat(self):
        return self._robot_entity.get_quat()

    @property
    def base_lin_vel(self) -> torch.Tensor:
        return self._robot_entity.get_vel()

    @property
    def base_ang_vel(self) -> torch.Tensor:
        return self._robot_entity.get_ang()

    @property
    def dof_pos(self) -> torch.Tensor:
        return self._robot_entity.get_dofs_position()[:, 6:]

    @property
    def dof_vel(self) -> torch.Tensor:
        return self._robot_entity.get_dofs_velocity()[:, 6:]

    @property
    def link_pos(self) -> torch.Tensor:
        return self._robot_entity.get_links_pos()

    @property
    def link_quat(self) -> torch.Tensor:
        return self._robot_entity.get_links_quat()

    @property
    def entity(self):
        return self._robot_entity

    @property
    def ref_entity(self):
        return self._ref_entity

    @property
    def non_root_joint_positions(self) -> torch.Tensor:
        # exclude pelvis root_joint
        pos = self._robot_entity.get_dofs_position()[:, self.non_root_joints]
        return pos

    @property
    def non_root_joint_velocities(self) -> torch.Tensor:
        return self._robot_entity.get_dofs_velocity()[
            :, self.non_root_joints
        ]  # exclude pelvis root_joint

    @property
    def joint_positions(self) -> torch.Tensor:
        pos = self._robot_entity.get_dofs_position()
        return pos

    @property
    def joint_velocities(self) -> torch.Tensor:
        return self._robot_entity.get_dofs_velocity()
