"""
MuJoCo Warp physics engine implementation.

This module wraps the MuJoCo Warp physics engine to conform to the BaseEngine interface.
Following a simple wrapper pattern similar to GenesisEngine.

Design:
- Single-phase initialization: Build complete model before MJWarp initialization
- MuJoCo native joint ordering: Simple, no complex remapping
- Direct API delegation: Minimal wrapper overhead
- Batch simulation via nworld parameter
"""

from typing import Optional, List, Dict, Tuple
import os
import tempfile
import torch
import mujoco
import mujoco_warp as mjw
import mujoco.viewer
import warp as wp
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

from add_gym.engine.base_engine import (
    BaseEngine,
    BaseScene,
    BaseEntity,
    BaseJoint,
    BaseLink,
    BaseCamera
)


class MJWarpCamera(BaseCamera):
    """Wrapper for MuJoCo Warp camera rendering."""

    def __init__(self, model, data, res: Tuple[int, int], pos: Tuple[float, float, float],
                 lookat: Tuple[float, float, float], fov: float):
        self._model = model
        self._data = data
        self._res = res
        self._recording = False
        self._frames = []

        # Create MuJoCo camera
        self._cam = mujoco.MjvCamera()
        self._cam.lookat = np.array(lookat)
        self._cam.distance = np.linalg.norm(np.array(pos) - np.array(lookat))

        # Compute azimuth and elevation from pos and lookat
        rel_pos = np.array(pos) - np.array(lookat)
        self._cam.azimuth = np.degrees(np.arctan2(rel_pos[1], rel_pos[0]))
        self._cam.elevation = np.degrees(np.arcsin(rel_pos[2] / self._cam.distance))

    def follow_entity(self, entity: BaseEntity) -> None:
        """Set camera to follow an entity."""
        # For now, update lookat to entity position
        # This is a simplified implementation
        if hasattr(entity, '_entity'):
            entity = entity._entity
        # TODO: Implement proper entity following
        pass

    def start_recording(self) -> None:
        """Start recording video."""
        self._recording = True
        self._frames = []

    def stop_recording(self, filename: str, fps: int = 30) -> None:
        """Stop recording and save video to file."""
        if not self._recording:
            return

        self._recording = False

        # Save frames to video using OpenCV or imageio
        try:
            import imageio
            imageio.mimsave(filename, self._frames, fps=fps)
        except ImportError:
            import cv2
            height, width = self._frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            for frame in self._frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()

        self._frames = []

    def render(self) -> None:
        """Render a single frame."""
        # TODO: Implement MuJoCo rendering
        # This requires setting up MjvScene and MjrContext
        # For now, this is a placeholder
        pass

    @property
    def _in_recording(self) -> bool:
        """Check if camera is currently recording."""
        return self._recording


class MJWarpLink(BaseLink):
    """Wrapper for MuJoCo Warp link/body."""

    def __init__(self, name: str, global_idx: int, local_idx: int):
        self._name = name
        self._global_idx = global_idx
        self._local_idx = local_idx

    @property
    def idx(self) -> int:
        """Global link index across all entities."""
        return self._global_idx

    @property
    def idx_local(self) -> int:
        """Local link index within the entity."""
        return self._local_idx

    @property
    def name(self) -> str:
        """Link name."""
        return self._name


class MJWarpJoint(BaseJoint):
    """Wrapper for MuJoCo Warp joint."""

    def __init__(self, name: str, global_dof_indices: List[int],
                 local_dof_indices: List[int], limits: List[Tuple[float, float]]):
        self._name = name
        self._global_dof_indices = global_dof_indices
        self._local_dof_indices = local_dof_indices
        self._limits = limits

    @property
    def dofs_idx(self) -> List[int]:
        """Global DOF indices for this joint."""
        return self._global_dof_indices

    @property
    def dofs_idx_local(self) -> List[int]:
        """Local DOF indices for this joint within the entity."""
        return self._local_dof_indices

    @property
    def dofs_limit(self) -> List[Tuple[float, float]]:
        """DOF limits as list of (lower, upper) tuples."""
        return self._limits

    @property
    def name(self) -> str:
        """Joint name."""
        return self._name


class MJWarpEntity(BaseEntity):
    """
    Wrapper for MuJoCo Warp entity.

    Stores references to the shared MuJoCo model/data and entity boundaries.
    All operations use direct slicing into MuJoCo arrays.
    """

    def __init__(self, entity_name: str):
        """Create entity stub (not yet built)."""
        self._name = entity_name
        self._mj_model = None  # Original mujoco.MjModel for metadata
        self._model = None     # MuJoCo Warp wrapped model
        self._data = None
        self._nworld = None
        self._device = None    # Torch device for tensors

        # Entity boundaries (set during build())
        self._body_start_idx = None
        self._body_count = None
        self._qpos_start_idx = None  # Start index in qpos array
        self._qpos_count = None      # Number of qpos elements (nq)
        self._dof_start_idx = None   # Start index in qvel array (same as in ctrl for actuators)
        self._dof_count = None       # Number of DOFs (nv)
        self._actuator_start_idx = None
        self._actuator_count = None

        # Metadata (set during build())
        self._joint_metadata = []
        self._link_metadata = []

        # Control state
        self._kp = None
        self._kv = None
        self._target_positions = None

    def _finalize(self, mj_model, model, data, nworld: int,
                  body_start: int, body_count: int,
                  qpos_start: int, qpos_count: int,
                  dof_start: int, dof_count: int,
                  actuator_start: int, actuator_count: int,
                  joint_metadata: List[Dict], link_metadata: List[Dict],
                  device: torch.device):
        """Finalize entity after model is built."""
        self._mj_model = mj_model  # Original mujoco.MjModel
        self._model = model        # MuJoCo Warp wrapped model
        self._data = data
        self._nworld = nworld
        self._device = device
        self._body_start_idx = body_start
        self._body_count = body_count
        self._qpos_start_idx = qpos_start
        self._qpos_count = qpos_count
        self._dof_start_idx = dof_start
        self._dof_count = dof_count
        self._actuator_start_idx = actuator_start
        self._actuator_count = actuator_count
        self._joint_metadata = joint_metadata
        self._link_metadata = link_metadata

        # Initialize control state
        self._kp = torch.zeros(dof_count, device=device)
        self._kv = torch.zeros(dof_count, device=device)
        self._target_positions = torch.zeros(nworld, dof_count, device=device)

    # ========== Position and Orientation ==========

    def get_pos(self) -> torch.Tensor:
        """Get entity base position."""
        # Root body position
        body_idx = self._body_start_idx
        xpos_np = self._data.xpos.numpy()  # Convert Warp array to numpy
        return torch.from_numpy(xpos_np[:, body_idx, :].copy()).to(self._device)

    def set_pos(self, pos: torch.Tensor, envs_idx: Optional[torch.Tensor] = None) -> None:
        """Set entity base position."""
        if envs_idx is None:
            envs_idx = slice(None)

        # For free-floating bodies, root position is in qpos
        # For fixed bodies, we need to set xpos directly
        # Simplified: assume free-floating root
        body_idx = self._body_start_idx
        if isinstance(pos, torch.Tensor):
            pos = pos.cpu().numpy()

        # Access Warp array and modify
        xpos_np = self._data.xpos.numpy()
        xpos_np[envs_idx, body_idx, :] = pos
        self._data.xpos.assign(xpos_np)

    def get_quat(self) -> torch.Tensor:
        """Get entity base orientation as quaternion."""
        body_idx = self._body_start_idx
        # MuJoCo quaternions are (w, x, y, z)
        xquat_np = self._data.xquat.numpy()
        return torch.from_numpy(xquat_np[:, body_idx, :].copy()).to(self._device)

    def set_quat(self, quat: torch.Tensor, envs_idx: Optional[torch.Tensor] = None) -> None:
        """Set entity base orientation as quaternion."""
        if envs_idx is None:
            envs_idx = slice(None)

        body_idx = self._body_start_idx
        if isinstance(quat, torch.Tensor):
            quat = quat.cpu().numpy()

        # Access Warp array and modify
        xquat_np = self._data.xquat.numpy()
        xquat_np[envs_idx, body_idx, :] = quat
        self._data.xquat.assign(xquat_np)

    # ========== Velocities ==========

    def get_vel(self) -> torch.Tensor:
        """Get entity base linear velocity."""
        # Body velocities in MuJoCo
        # For simplicity, get from qvel for free-floating base
        if self._dof_start_idx is not None and self._dof_count >= 3:
            qvel_np = self._data.qvel.numpy()
            return torch.from_numpy(qvel_np[:, self._dof_start_idx:self._dof_start_idx+3].copy()).to(self._device)
        return torch.zeros(self._nworld, 3, device=self._device)

    def get_ang(self) -> torch.Tensor:
        """Get entity base angular velocity."""
        # Angular velocity from qvel for free-floating base
        if self._dof_start_idx is not None and self._dof_count >= 6:
            qvel_np = self._data.qvel.numpy()
            return torch.from_numpy(qvel_np[:, self._dof_start_idx+3:self._dof_start_idx+6].copy()).to(self._device)
        return torch.zeros(self._nworld, 3, device=self._device)

    # ========== Degrees of Freedom (DOFs) ==========

    def get_dofs_position(self) -> torch.Tensor:
        """Get all DOF positions for the entity."""
        start = self._dof_start_idx
        end = start + self._dof_count
        qpos_np = self._data.qpos.numpy()
        return torch.from_numpy(qpos_np[:, start:end].copy()).to(self._device)

    def set_dofs_position(
        self,
        position: torch.Tensor,
        envs_idx: Optional[torch.Tensor] = None,
        dofs_idx_local: Optional[List[int]] = None
    ) -> None:
        """Set DOF positions for the entity."""
        if envs_idx is None:
            envs_idx = slice(None)

        start = self._dof_start_idx

        if isinstance(position, torch.Tensor):
            position = position.cpu().numpy()

        qpos_np = self._data.qpos.numpy()
        if dofs_idx_local is None:
            qpos_np[envs_idx, start:start+self._dof_count] = position
        else:
            for i, local_idx in enumerate(dofs_idx_local):
                qpos_np[envs_idx, start + local_idx] = position[:, i]
        self._data.qpos.assign(qpos_np)

    def get_dofs_velocity(self) -> torch.Tensor:
        """Get all DOF velocities for the entity."""
        start = self._dof_start_idx
        end = start + self._dof_count
        qvel_np = self._data.qvel.numpy()
        return torch.from_numpy(qvel_np[:, start:end].copy()).to(self._device)

    def set_dofs_velocity(
        self,
        velocity: torch.Tensor,
        envs_idx: Optional[torch.Tensor] = None
    ) -> None:
        """Set DOF velocities for the entity."""
        if envs_idx is None:
            envs_idx = slice(None)

        start = self._dof_start_idx

        if isinstance(velocity, torch.Tensor):
            velocity = velocity.cpu().numpy()

        qvel_np = self._data.qvel.numpy()
        qvel_np[envs_idx, start:start+self._dof_count] = velocity
        self._data.qvel.assign(qvel_np)

    def control_dofs_position(
        self,
        position: torch.Tensor,
        dofs_idx_local: Optional[List[int]] = None
    ) -> None:
        """Set target positions for DOF position control."""
        if dofs_idx_local is None:
            dofs_idx_local = list(range(self._dof_count))

        # Store target positions
        for i, local_idx in enumerate(dofs_idx_local):
            self._target_positions[:, local_idx] = position[:, i]

        # Get current state
        qpos_np = self._data.qpos.numpy()
        qvel_np = self._data.qvel.numpy()
        ctrl_np = self._data.ctrl.numpy()

        # Compute PD control
        for local_idx in dofs_idx_local:
            dof_idx = self._dof_start_idx + local_idx
            actuator_idx = self._actuator_start_idx + local_idx

            if actuator_idx >= self._actuator_start_idx + self._actuator_count:
                continue

            # PD control: ctrl = kp * (target - current) - kv * velocity
            current_pos = qpos_np[:, dof_idx]
            current_vel = qvel_np[:, dof_idx]
            target_pos = self._target_positions[:, local_idx].cpu().numpy()

            error = target_pos - current_pos
            ctrl_np[:, actuator_idx] = self._kp[local_idx].item() * error - self._kv[local_idx].item() * current_vel

        self._data.ctrl.assign(ctrl_np)

    def set_dofs_kp(self, kp: torch.Tensor) -> None:
        """Set position control gains (stiffness)."""
        self._kp = kp.clone()

    def set_dofs_kv(self, kv: torch.Tensor) -> None:
        """Set velocity control gains (damping)."""
        self._kv = kv.clone()

    def zero_all_dofs_velocity(self, envs_idx: Optional[torch.Tensor] = None) -> None:
        """Set all DOF velocities to zero."""
        if envs_idx is None:
            envs_idx = slice(None)

        start = self._dof_start_idx
        qvel_np = self._data.qvel.numpy()
        qvel_np[envs_idx, start:start+self._dof_count] = 0.0
        self._data.qvel.assign(qvel_np)

    # ========== Links ==========

    def get_links_pos(self) -> torch.Tensor:
        """Get positions of all links."""
        start = self._body_start_idx
        end = start + self._body_count
        xpos_np = self._data.xpos.numpy()
        return torch.from_numpy(xpos_np[:, start:end, :].copy()).to(self._device)

    def get_links_quat(self) -> torch.Tensor:
        """Get orientations of all links as quaternions."""
        start = self._body_start_idx
        end = start + self._body_count
        xquat_np = self._data.xquat.numpy()
        return torch.from_numpy(xquat_np[:, start:end, :].copy()).to(self._device)

    def get_links_net_contact_force(self) -> torch.Tensor:
        """Get net contact forces on all links."""
        # MuJoCo stores contact forces in cfrc_ext
        start = self._body_start_idx
        end = start + self._body_count
        # cfrc_ext has 6 components (3 force + 3 torque)
        cfrc_ext_np = self._data.cfrc_ext.numpy()
        return torch.from_numpy(cfrc_ext_np[:, start:end, :3].copy()).to(self._device)

    # ========== Contacts ==========

    def _is_my_geom(self, geom_id: int, mj_model) -> bool:
        """Check if a geometry belongs to this entity."""
        # Get body ID for this geom
        if geom_id < 0 or geom_id >= mj_model.ngeom:
            return False

        body_id = mj_model.geom_bodyid[geom_id]

        # Check if body is in our range
        return self._body_start_idx <= body_id < self._body_start_idx + self._body_count

    def get_contacts(
        self,
        with_entity: "BaseEntity",
        exclude_self_contact: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Get contact information with another entity."""
        # Unwrap if wrapped
        if isinstance(with_entity, MJWarpEntity):
            other_entity = with_entity
        else:
            other_entity = with_entity

        # Use the original MuJoCo model for metadata
        mj_model = self._mj_model

        # Lists to store contact data for each world
        all_link_a = []
        all_link_b = []
        all_valid = []

        # Iterate through each world
        for world_idx in range(self._nworld):
            link_a = []
            link_b = []
            valid = []

            # Access contact data for this world
            # Note: MuJoCo Warp batches contacts, so we need to index by world
            # The exact structure depends on MuJoCo Warp implementation
            try:
                ncon = self._data.ncon[world_idx] if hasattr(self._data.ncon, '__getitem__') else self._data.ncon

                for i in range(min(int(ncon), 100)):  # Limit to avoid excessive contacts
                    # Get contact geom IDs
                    geom1 = self._data.contact.geom1[world_idx, i] if hasattr(self._data.contact.geom1, '__getitem__') else self._data.contact[i].geom1
                    geom2 = self._data.contact.geom2[world_idx, i] if hasattr(self._data.contact.geom2, '__getitem__') else self._data.contact[i].geom2

                    # Check if contact involves both entities
                    geom1_is_mine = self._is_my_geom(int(geom1), mj_model)
                    geom2_is_mine = self._is_my_geom(int(geom2), mj_model)
                    geom1_is_other = other_entity._is_my_geom(int(geom1), mj_model)
                    geom2_is_other = other_entity._is_my_geom(int(geom2), mj_model)

                    if (geom1_is_mine and geom2_is_other) or (geom2_is_mine and geom1_is_other):
                        # Get body IDs
                        body1 = mj_model.geom_bodyid[int(geom1)]
                        body2 = mj_model.geom_bodyid[int(geom2)]

                        # Convert to local indices
                        if geom1_is_mine:
                            local_a = body1 - self._body_start_idx
                            local_b = body2 - other_entity._body_start_idx
                        else:
                            local_a = body2 - self._body_start_idx
                            local_b = body1 - other_entity._body_start_idx

                        link_a.append(local_a)
                        link_b.append(local_b)
                        valid.append(True)

            except (AttributeError, IndexError):
                # MuJoCo Warp contact structure might be different
                # Fall back to no contacts
                pass

            all_link_a.append(link_a)
            all_link_b.append(link_b)
            all_valid.append(valid)

        # Convert to tensors
        # For now, return flattened contact data
        # TODO: Better batching structure
        flat_link_a = []
        flat_link_b = []
        flat_valid = []

        for link_a, link_b, valid in zip(all_link_a, all_link_b, all_valid):
            flat_link_a.extend(link_a)
            flat_link_b.extend(link_b)
            flat_valid.extend(valid)

        return {
            'link_a': torch.tensor(flat_link_a) if flat_link_a else torch.empty(0, dtype=torch.long),
            'link_b': torch.tensor(flat_link_b) if flat_link_b else torch.empty(0, dtype=torch.long),
            'valid_mask': torch.tensor(flat_valid) if flat_valid else torch.empty(0, dtype=torch.bool)
        }

        # Lists to store contact data for each world
        all_link_a = []
        all_link_b = []
        all_valid = []

        # Iterate through each world
        for world_idx in range(self._nworld):
            link_a = []
            link_b = []
            valid = []

            # Access contact data for this world
            # Note: MuJoCo Warp batches contacts, so we need to index by world
            # The exact structure depends on MuJoCo Warp implementation
            try:
                ncon = self._data.ncon[world_idx] if hasattr(self._data.ncon, '__getitem__') else self._data.ncon

                for i in range(min(ncon, 100)):  # Limit to avoid excessive contacts
                    # Get contact geom IDs
                    geom1 = self._data.contact.geom1[world_idx, i] if hasattr(self._data.contact.geom1, '__getitem__') else self._data.contact[i].geom1
                    geom2 = self._data.contact.geom2[world_idx, i] if hasattr(self._data.contact.geom2, '__getitem__') else self._data.contact[i].geom2

                    # Check if contact involves both entities
                    geom1_is_mine = self._is_my_geom(int(geom1))
                    geom2_is_mine = self._is_my_geom(int(geom2))
                    geom1_is_other = other_entity._is_my_geom(int(geom1))
                    geom2_is_other = other_entity._is_my_geom(int(geom2))

                    if (geom1_is_mine and geom2_is_other) or (geom2_is_mine and geom1_is_other):
                        # Get body IDs
                        body1 = mj_model.geom_bodyid[int(geom1)]
                        body2 = mj_model.geom_bodyid[int(geom2)]

                        # Convert to local indices
                        if geom1_is_mine:
                            local_a = body1 - self._body_start_idx
                            local_b = body2 - other_entity._body_start_idx
                        else:
                            local_a = body2 - self._body_start_idx
                            local_b = body1 - other_entity._body_start_idx

                        link_a.append(local_a)
                        link_b.append(local_b)
                        valid.append(True)

            except (AttributeError, IndexError):
                # MuJoCo Warp contact structure might be different
                # Fall back to no contacts
                pass

            all_link_a.append(link_a)
            all_link_b.append(link_b)
            all_valid.append(valid)

        # Convert to tensors
        # For now, return flattened contact data
        # TODO: Better batching structure
        flat_link_a = []
        flat_link_b = []
        flat_valid = []

        for link_a, link_b, valid in zip(all_link_a, all_link_b, all_valid):
            flat_link_a.extend(link_a)
            flat_link_b.extend(link_b)
            flat_valid.extend(valid)

        return {
            'link_a': torch.tensor(flat_link_a) if flat_link_a else torch.empty(0, dtype=torch.long),
            'link_b': torch.tensor(flat_link_b) if flat_link_b else torch.empty(0, dtype=torch.long),
            'valid_mask': torch.tensor(flat_valid) if flat_valid else torch.empty(0, dtype=torch.bool)
        }

    # ========== Geometry ==========

    def get_AABB(self) -> torch.Tensor:
        """Get axis-aligned bounding box."""
        # Compute AABB from body positions
        link_pos = self.get_links_pos()  # (nworld, n_links, 3)

        # Simple AABB: min and max across all links
        # TODO: Use actual geom sizes for better AABB
        min_pos = torch.min(link_pos, dim=1)[0]  # (nworld, 3)
        max_pos = torch.max(link_pos, dim=1)[0]  # (nworld, 3)

        # Stack to (nworld, 2, 3)
        return torch.stack([min_pos, max_pos], dim=1)

    # ========== Structure Access ==========

    def get_joint(self, name: str) -> BaseJoint:
        """Get joint by name."""
        for joint_info in self._joint_metadata:
            if joint_info['name'] == name:
                return MJWarpJoint(
                    name=joint_info['name'],
                    global_dof_indices=joint_info['global_dof_indices'],
                    local_dof_indices=joint_info['local_dof_indices'],
                    limits=joint_info['limits']
                )
        raise ValueError(f"Joint '{name}' not found in entity '{self._name}'")

    def get_link(self, name: str) -> BaseLink:
        """Get link by name."""
        for link_info in self._link_metadata:
            if link_info['name'] == name:
                return MJWarpLink(
                    name=link_info['name'],
                    global_idx=link_info['global_idx'],
                    local_idx=link_info['local_idx']
                )
        raise ValueError(f"Link '{name}' not found in entity '{self._name}'")

    @property
    def joints(self) -> List[BaseJoint]:
        """List of all joints in the entity."""
        return [
            MJWarpJoint(
                name=j['name'],
                global_dof_indices=j['global_dof_indices'],
                local_dof_indices=j['local_dof_indices'],
                limits=j['limits']
            )
            for j in self._joint_metadata
        ]

    @property
    def links(self) -> List[BaseLink]:
        """List of all links in the entity."""
        return [
            MJWarpLink(
                name=l['name'],
                global_idx=l['global_idx'],
                local_idx=l['local_idx']
            )
            for l in self._link_metadata
        ]

    @property
    def n_dofs(self) -> int:
        """Total number of degrees of freedom."""
        return self._dof_count

    # ========== Special Methods ==========

    def set_qpos(self, qpos: torch.Tensor, envs_idx: Optional[torch.Tensor] = None) -> None:
        """Set generalized position (qpos)."""
        if envs_idx is None:
            envs_idx = slice(None)

        if isinstance(qpos, torch.Tensor):
            qpos = qpos.cpu().numpy()

        start = self._qpos_start_idx
        qpos_np = self._data.qpos.numpy()
        qpos_np[envs_idx, start:start+self._qpos_count] = qpos
        self._data.qpos.assign(qpos_np)


class MJWarpScene(BaseScene):
    """Wrapper for MuJoCo Warp scene."""

    def __init__(self, sim_options: Dict, rigid_options: Dict, viewer_enabled: bool = True):
        self._sim_options = sim_options
        self._rigid_options = rigid_options
        self._viewer_enabled = viewer_enabled

        # Entities to be built
        self._entities = []
        self._entity_configs = []

        # Built scene state
        self._mj_model = None  # Original mujoco.MjModel for metadata
        self._model = None     # MuJoCo Warp wrapped model
        self._data = None
        self._nworld = None
        self._timestep = 0
        
        self._viewer: Optional[mujoco.viewer.Handle] = None

    def add_entity(
        self,
        morph_type: str,
        morph_file: Optional[str] = None,
        morph_pos: Optional[Tuple[float, float, float]] = None,
        morph_quat: Optional[Tuple[float, float, float, float]] = None,
        material_type: str = "rigid",
        visualize_contact: bool = True
    ) -> BaseEntity:
        """Add an entity to the scene (not yet built)."""
        entity_name = f"entity_{len(self._entities)}"
        entity = MJWarpEntity(entity_name)

        config = {
            'entity': entity,
            'morph_type': morph_type,
            'morph_file': morph_file,
            'morph_pos': morph_pos or (0.0, 0.0, 0.0),
            'morph_quat': morph_quat or (1.0, 0.0, 0.0, 0.0),
            'material_type': material_type,
            'visualize_contact': visualize_contact
        }

        self._entities.append(entity)
        self._entity_configs.append(config)

        return entity

    def add_camera(
        self,
        res: Tuple[int, int],
        pos: Tuple[float, float, float],
        lookat: Tuple[float, float, float],
        fov: float
    ) -> BaseCamera:
        """Add a camera to the scene."""
        # Camera requires model/data, so defer creation until after build()
        camera = MJWarpCamera(self._model, self._data, res, pos, lookat, fov)
        return camera

    def build(self, n_envs: int, env_spacing: Tuple[float, float]) -> None:
        """Build the scene with multiple parallel environments."""
        self._nworld = n_envs

        # Step 1: Load MuJoCo model with proper directory context
        self._mj_model = self._load_composite_model()
        mj_data = mujoco.MjData(self._mj_model)

        # Step 2: Put model into MuJoCo Warp
        self._model = mjw.put_model(self._mj_model)

        # Step 3: Create MuJoCo Warp data
        # TODO: Determine appropriate nconmax and njmax
        nconmax = 1000  # Maximum contacts per world
        njmax = 500     # Maximum constraints per world

        self._data = mjw.put_data(self._mj_model, mj_data, nworld=n_envs, nconmax=nconmax, njmax=njmax)

        # Step 4: Extract metadata and finalize entities
        self._extract_entity_metadata()
        
        if self._viewer_enabled:
            self._viewer = mujoco.viewer.launch_passive(self._mj_model, mj_data)

    def _get_device(self):
        """Get the device from the parent scene/engine."""
        # This is a bit of a hack - we should pass device more explicitly
        # For now, check if we're on GPU based on the Warp array device
        if hasattr(self._data.qpos, 'device'):
            if 'cuda' in str(self._data.qpos.device):
                return torch.device('cuda')
        return torch.device('cpu')

    def _load_composite_model(self) -> mujoco.MjModel:
        """Load composite MuJoCo model with proper directory context for assets."""
        # Simple case: single robot entity
        robot_configs = [c for c in self._entity_configs if c['morph_type'] in ['urdf', 'mjcf']]
        plane_configs = [c for c in self._entity_configs if c['morph_type'] == 'plane']

        if len(robot_configs) == 0:
            # No robot, just create empty scene with plane
            root = ET.Element('mujoco')
            root.set('model', 'composite_scene')

            compiler = ET.SubElement(root, 'compiler')
            compiler.set('angle', 'radian')

            worldbody = ET.SubElement(root, 'worldbody')

            # Add plane if present
            if plane_configs:
                geom = ET.SubElement(worldbody, 'geom')
                geom.set('type', 'plane')
                geom.set('size', '10 10 0.1')
                geom.set('rgba', '0.8 0.8 0.8 1')

            xml_str = ET.tostring(root, encoding='unicode')
            return mujoco.MjModel.from_xml_string(xml_str)

        elif len(robot_configs) == 1:
            # Single robot: load file with proper directory context
            config = robot_configs[0]
            morph_file = config['morph_file']

            if config['morph_type'] == 'urdf':
                # Ignore URDF for now
                raise NotImplementedError("URDF loading not yet implemented")

            elif config['morph_type'] == 'mjcf':
                if not plane_configs:
                    # No modifications needed - load directly
                    # This preserves the directory context for mesh files
                    return mujoco.MjModel.from_xml_path(morph_file)
                else:
                    # Need to add plane - modify XML and save to temp file in same directory
                    tree = ET.parse(morph_file)
                    root = tree.getroot()

                    worldbody = root.find('worldbody')
                    if worldbody is None:
                        worldbody = ET.SubElement(root, 'worldbody')

                    # Add plane at the beginning of worldbody
                    geom = ET.Element('geom')
                    geom.set('type', 'plane')
                    geom.set('size', '10 10 0.1')
                    geom.set('rgba', '0.8 0.8 0.8 1')
                    worldbody.insert(0, geom)

                    # Write to temporary file in the same directory as original
                    morph_dir = os.path.dirname(os.path.abspath(morph_file))
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', dir=morph_dir, delete=False) as f:
                        temp_path = f.name
                        tree.write(f, encoding='unicode')

                    try:
                        # Load from temp file (preserves directory context)
                        model = mujoco.MjModel.from_xml_path(temp_path)
                    finally:
                        # Clean up temp file
                        os.unlink(temp_path)

                    return model

        else:
            # Multiple robots: need proper merging
            # TODO: Implement multi-robot merging
            raise NotImplementedError("Multiple robot entities not yet supported")

    def _extract_entity_metadata(self):
        """Extract metadata for each entity from the built model."""
        # For now, assume single entity or simple entity boundaries
        # In a full implementation, we'd need to track entity boundaries during model composition

        # Use the original MuJoCo model for metadata (has name_jntadr, etc.)
        mj_model = self._mj_model

        # Get device for tensors
        device = self._get_device()

        body_offset = 0
        qpos_offset = 0
        dof_offset = 0
        actuator_offset = 0

        for i, (entity, config) in enumerate(zip(self._entities, self._entity_configs)):
            if config['morph_type'] == 'plane':
                # Plane has 1 body, 0 qpos, 0 DOFs, 0 actuators
                body_count = 1
                qpos_count = 0
                dof_count = 0
                actuator_count = 0

                # Finalize with no joints or actuators
                entity._finalize(
                    mj_model=self._mj_model,
                    model=self._model,
                    data=self._data,
                    nworld=self._nworld,
                    body_start=body_offset,
                    body_count=body_count,
                    qpos_start=qpos_offset,
                    qpos_count=qpos_count,
                    dof_start=dof_offset,
                    dof_count=dof_count,
                    actuator_start=actuator_offset,
                    actuator_count=actuator_count,
                    joint_metadata=[],
                    link_metadata=[{
                        'name': 'plane',
                        'global_idx': body_offset,
                        'local_idx': 0
                    }],
                    device=device
                )

            else:
                # For URDF/MJCF entities, extract from model
                # In a simple case with one robot, use all remaining bodies/dofs
                body_count = mj_model.nbody - body_offset
                qpos_count = mj_model.nq - qpos_offset
                dof_count = mj_model.nv - dof_offset
                actuator_count = mj_model.nu - actuator_offset

                # Extract joint metadata
                joint_metadata = []
                for jnt_id in range(mj_model.njnt):
                    # Get joint name
                    jnt_name_start = mj_model.name_jntadr[jnt_id]
                    jnt_name_end = mj_model.names[jnt_name_start:].find(b'\x00')
                    jnt_name = mj_model.names[jnt_name_start:jnt_name_start+jnt_name_end].decode('utf-8')

                    # Get DOF address and count for this joint
                    dof_adr = mj_model.jnt_dofadr[jnt_id]
                    jnt_type = mj_model.jnt_type[jnt_id]

                    # Determine number of DOFs based on joint type
                    if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                        ndof = 6
                    elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                        ndof = 3
                    elif jnt_type == mujoco.mjtJoint.mjJNT_SLIDE or jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
                        ndof = 1
                    else:
                        ndof = 1  # Default

                    # Get DOF indices (both global and local)
                    global_dof_indices = list(range(dof_adr, dof_adr + ndof))
                    local_dof_indices = [idx - dof_offset for idx in global_dof_indices if idx >= dof_offset and idx < dof_offset + dof_count]

                    if not local_dof_indices:
                        continue

                    # Get joint limits
                    limits = []
                    for dof_idx in global_dof_indices:
                        if dof_idx < mj_model.nv:
                            jnt_limited = mj_model.jnt_limited[jnt_id]
                            if jnt_limited:
                                jnt_range = mj_model.jnt_range[jnt_id]
                                limits.append((float(jnt_range[0]), float(jnt_range[1])))
                            else:
                                limits.append((-np.inf, np.inf))

                    joint_metadata.append({
                        'name': jnt_name,
                        'global_dof_indices': global_dof_indices,
                        'local_dof_indices': local_dof_indices,
                        'limits': limits
                    })

                # Extract link/body metadata
                link_metadata = []
                for body_id in range(body_offset, body_offset + body_count):
                    if body_id >= mj_model.nbody:
                        break

                    # Get body name
                    body_name_start = mj_model.name_bodyadr[body_id]
                    body_name_end = mj_model.names[body_name_start:].find(b'\x00')
                    body_name = mj_model.names[body_name_start:body_name_start+body_name_end].decode('utf-8')

                    link_metadata.append({
                        'name': body_name if body_name else f'body_{body_id}',
                        'global_idx': body_id,
                        'local_idx': body_id - body_offset
                    })

                # Finalize entity
                entity._finalize(
                    mj_model=self._mj_model,
                    model=self._model,
                    data=self._data,
                    nworld=self._nworld,
                    body_start=body_offset,
                    body_count=body_count,
                    qpos_start=qpos_offset,
                    qpos_count=qpos_count,
                    dof_start=dof_offset,
                    dof_count=dof_count,
                    actuator_start=actuator_offset,
                    actuator_count=actuator_count,
                    joint_metadata=joint_metadata,
                    link_metadata=link_metadata,
                    device=device
                )

            body_offset += body_count
            qpos_offset += qpos_count
            dof_offset += dof_count
            actuator_offset += actuator_count

    def step(self) -> None:
        """Step the simulation forward by one timestep."""
        mjw.forward(self._model, self._data)
        mjw.kinematics(self._model, self._data)
        mjw.step(self._model, self._data)
        self._timestep += 1
        if self._viewer_enabled:
            self._viewer.sync()

    @property
    def t(self) -> int:
        """Current simulation timestep."""
        return self._timestep


class MJWarpEngine(BaseEngine):
    """MuJoCo Warp physics engine implementation."""

    def __init__(self, **kwargs):
        """Initialize MuJoCo Warp engine.

        Args:
            **kwargs: Configuration parameters from Hydra.
        """
        self._device = None
        self._tc_float = None
        self._config = kwargs

    def init(self, backend: str, precision: str) -> None:
        """Initialize the MuJoCo Warp engine.

        Args:
            backend: Backend type - "gpu" or "cpu"
            precision: Numerical precision - "32" or "64"
        """
        # Initialize Warp
        if backend == "gpu":
            wp.init()
            self._device = torch.device("cuda")
        elif backend == "cpu":
            wp.init()
            self._device = torch.device("cpu")
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Set precision
        if precision == "32":
            self._tc_float = torch.float32
        elif precision == "64":
            self._tc_float = torch.float64
        else:
            raise ValueError(f"Unknown precision: {precision}")

    def create_scene(
        self,
        show_viewer: bool,
        sim_options: Dict,
        rigid_options: Dict,
        vis_options: Optional[Dict] = None,
        viewer_options: Optional[Dict] = None
    ) -> BaseScene:
        """Create a simulation scene.

        Args:
            show_viewer: Whether to show visualization window
            sim_options: Simulation options (dt, etc.)
            rigid_options: Rigid body options (solver, collision, etc.)
            vis_options: Visualization options
            viewer_options: Viewer window options

        Returns:
            Created scene instance
        """
        # Create and return scene
        scene = MJWarpScene(sim_options=sim_options, rigid_options=rigid_options, viewer_enabled=show_viewer)
        return scene

    @property
    def device(self) -> torch.device:
        """PyTorch device for tensor operations."""
        if self._device is None:
            raise RuntimeError("Engine not initialized. Call init() first.")
        return self._device

    @property
    def tc_float(self) -> torch.dtype:
        """PyTorch tensor float dtype (precision)."""
        if self._tc_float is None:
            raise RuntimeError("Engine not initialized. Call init() first.")
        return self._tc_float
