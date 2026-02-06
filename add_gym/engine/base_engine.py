"""
Abstract base classes for physics engine abstraction layer.

This module defines the interface that all physics engines (Genesis, MuJoCo Warp, etc.)
must implement to be compatible with the training framework.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple
import torch


class BaseCamera(ABC):
    """Abstract base class for camera rendering."""

    @abstractmethod
    def follow_entity(self, entity: "BaseEntity") -> None:
        """Set camera to follow an entity."""
        pass

    @abstractmethod
    def start_recording(self) -> None:
        """Start recording video."""
        pass

    @abstractmethod
    def stop_recording(self, filename: str, fps: int = 30) -> None:
        """Stop recording and save video to file."""
        pass

    @abstractmethod
    def render(self) -> None:
        """Render a single frame."""
        pass

    @property
    @abstractmethod
    def _in_recording(self) -> bool:
        """Check if camera is currently recording."""
        pass


class BaseLink(ABC):
    """Abstract base class for robot links."""

    @property
    @abstractmethod
    def idx(self) -> int:
        """Global link index across all entities."""
        pass

    @property
    @abstractmethod
    def idx_local(self) -> int:
        """Local link index within the entity."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Link name."""
        pass


class BaseJoint(ABC):
    """Abstract base class for robot joints."""

    @property
    @abstractmethod
    def dofs_idx(self) -> List[int]:
        """Global DOF indices for this joint."""
        pass

    @property
    @abstractmethod
    def dofs_idx_local(self) -> List[int]:
        """Local DOF indices for this joint within the entity."""
        pass

    @property
    @abstractmethod
    def dofs_limit(self) -> List[Tuple[float, float]]:
        """DOF limits as list of (lower, upper) tuples."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Joint name."""
        pass


class BaseEntity(ABC):
    """
    Abstract base class for physics entities (robots, objects, planes).

    All methods that accept tensors should handle batched operations across
    multiple environments (first dimension is batch/environment index).
    """

    # ========== Position and Orientation ==========

    @abstractmethod
    def get_pos(self) -> torch.Tensor:
        """
        Get entity base position.

        Returns:
            Tensor of shape [batch, 3] - XYZ positions
        """
        pass

    @abstractmethod
    def set_pos(self, pos: torch.Tensor, envs_idx: Optional[torch.Tensor] = None) -> None:
        """
        Set entity base position.

        Args:
            pos: Tensor of shape [batch, 3] - XYZ positions
            envs_idx: Optional environment indices to update (default: all)
        """
        pass

    @abstractmethod
    def get_quat(self) -> torch.Tensor:
        """
        Get entity base orientation as quaternion.

        Returns:
            Tensor of shape [batch, 4] - Quaternions (w, x, y, z)
        """
        pass

    @abstractmethod
    def set_quat(self, quat: torch.Tensor, envs_idx: Optional[torch.Tensor] = None) -> None:
        """
        Set entity base orientation as quaternion.

        Args:
            quat: Tensor of shape [batch, 4] - Quaternions (w, x, y, z)
            envs_idx: Optional environment indices to update (default: all)
        """
        pass

    # ========== Velocities ==========

    @abstractmethod
    def get_vel(self) -> torch.Tensor:
        """
        Get entity base linear velocity.

        Returns:
            Tensor of shape [batch, 3] - Linear velocities
        """
        pass

    @abstractmethod
    def get_ang(self) -> torch.Tensor:
        """
        Get entity base angular velocity.

        Returns:
            Tensor of shape [batch, 3] - Angular velocities
        """
        pass

    # ========== Degrees of Freedom (DOFs) ==========

    @abstractmethod
    def get_dofs_position(self) -> torch.Tensor:
        """
        Get all DOF positions for the entity.

        Returns:
            Tensor of shape [batch, n_dofs] - DOF positions
        """
        pass

    @abstractmethod
    def set_dofs_position(
        self,
        position: torch.Tensor,
        envs_idx: Optional[torch.Tensor] = None,
        dofs_idx_local: Optional[List[int]] = None
    ) -> None:
        """
        Set DOF positions for the entity.

        Args:
            position: Tensor of DOF positions
            envs_idx: Optional environment indices to update (default: all)
            dofs_idx_local: Optional local DOF indices to update (default: all)
        """
        pass

    @abstractmethod
    def get_dofs_velocity(self) -> torch.Tensor:
        """
        Get all DOF velocities for the entity.

        Returns:
            Tensor of shape [batch, n_dofs] - DOF velocities
        """
        pass

    @abstractmethod
    def set_dofs_velocity(
        self,
        velocity: torch.Tensor,
        envs_idx: Optional[torch.Tensor] = None
    ) -> None:
        """
        Set DOF velocities for the entity.

        Args:
            velocity: Tensor of DOF velocities
            envs_idx: Optional environment indices to update (default: all)
        """
        pass

    @abstractmethod
    def control_dofs_position(
        self,
        position: torch.Tensor,
        dofs_idx_local: Optional[List[int]] = None
    ) -> None:
        """
        Set target positions for DOF position control.

        Args:
            position: Target DOF positions
            dofs_idx_local: Optional local DOF indices to control (default: all)
        """
        pass

    @abstractmethod
    def set_dofs_kp(self, kp: torch.Tensor) -> None:
        """
        Set position control gains (stiffness).

        Args:
            kp: Tensor of Kp gains for each DOF
        """
        pass

    @abstractmethod
    def set_dofs_kv(self, kv: torch.Tensor) -> None:
        """
        Set velocity control gains (damping).

        Args:
            kv: Tensor of Kv gains for each DOF
        """
        pass

    @abstractmethod
    def zero_all_dofs_velocity(self, envs_idx: Optional[torch.Tensor] = None) -> None:
        """
        Set all DOF velocities to zero.

        Args:
            envs_idx: Optional environment indices to update (default: all)
        """
        pass

    # ========== Links ==========

    @abstractmethod
    def get_links_pos(self) -> torch.Tensor:
        """
        Get positions of all links.

        Returns:
            Tensor of shape [batch, n_links, 3] - Link positions
        """
        pass

    @abstractmethod
    def get_links_quat(self) -> torch.Tensor:
        """
        Get orientations of all links as quaternions.

        Returns:
            Tensor of shape [batch, n_links, 4] - Link quaternions
        """
        pass

    @abstractmethod
    def get_links_net_contact_force(self) -> torch.Tensor:
        """
        Get net contact forces on all links.

        Returns:
            Tensor of shape [batch, n_links, 3] - Contact forces
        """
        pass

    # ========== Contacts ==========

    @abstractmethod
    def get_contacts(
        self,
        with_entity: "BaseEntity",
        exclude_self_contact: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Get contact information with another entity.

        Args:
            with_entity: Entity to check contacts with
            exclude_self_contact: Whether to exclude self-collisions

        Returns:
            Dictionary with keys:
                'link_a': Link indices from this entity
                'link_b': Link indices from other entity
                'valid_mask': Boolean mask of valid contacts
        """
        pass

    # ========== Geometry ==========

    @abstractmethod
    def get_AABB(self) -> torch.Tensor:
        """
        Get axis-aligned bounding box.

        Returns:
            Tensor of shape [batch, 2, 3] - Min and max corners
        """
        pass

    # ========== Structure Access ==========

    @abstractmethod
    def get_joint(self, name: str) -> BaseJoint:
        """Get joint by name."""
        pass

    @abstractmethod
    def get_link(self, name: str) -> BaseLink:
        """Get link by name."""
        pass

    @property
    @abstractmethod
    def joints(self) -> List[BaseJoint]:
        """List of all joints in the entity."""
        pass

    @property
    @abstractmethod
    def links(self) -> List[BaseLink]:
        """List of all links in the entity."""
        pass

    @property
    @abstractmethod
    def n_dofs(self) -> int:
        """Total number of degrees of freedom."""
        pass

    # ========== Special Methods ==========

    @abstractmethod
    def set_qpos(self, qpos: torch.Tensor, envs_idx: Optional[torch.Tensor] = None) -> None:
        """
        Set generalized position (qpos) - includes root position, orientation, and DOFs.

        This is a convenience method used in some observation/visualization code.

        Args:
            qpos: Generalized position tensor
            envs_idx: Optional environment indices to update (default: all)
        """
        pass


class BaseScene(ABC):
    """
    Abstract base class for physics simulation scenes.

    A scene contains entities, manages simulation stepping, and provides
    rendering capabilities.
    """

    @abstractmethod
    def add_entity(
        self,
        morph_type: str,
        morph_file: Optional[str] = None,
        morph_pos: Optional[Tuple[float, float, float]] = None,
        morph_quat: Optional[Tuple[float, float, float, float]] = None,
        material_type: str = "rigid",
        visualize_contact: bool = True
    ) -> BaseEntity:
        """
        Add an entity to the scene.

        Args:
            morph_type: Type of entity - "urdf", "mjcf", or "plane"
            morph_file: Path to URDF/MJCF file (required for urdf/mjcf types)
            morph_pos: Initial position (x, y, z)
            morph_quat: Initial quaternion (w, x, y, z)
            material_type: Material type (default: "rigid")
            visualize_contact: Whether to visualize contacts for this entity

        Returns:
            Created entity instance
        """
        pass

    @abstractmethod
    def add_camera(
        self,
        res: Tuple[int, int],
        pos: Tuple[float, float, float],
        lookat: Tuple[float, float, float],
        fov: float
    ) -> BaseCamera:
        """
        Add a camera to the scene.

        Args:
            res: Resolution (width, height)
            pos: Camera position (x, y, z)
            lookat: Look-at target (x, y, z)
            fov: Field of view in degrees

        Returns:
            Created camera instance
        """
        pass

    @abstractmethod
    def build(self, n_envs: int, env_spacing: Tuple[float, float]) -> None:
        """
        Build the scene with multiple parallel environments.

        Args:
            n_envs: Number of parallel environments
            env_spacing: Spacing between environments (x, y)
        """
        pass

    @abstractmethod
    def step(self) -> None:
        """Step the simulation forward by one timestep."""
        pass

    @property
    @abstractmethod
    def t(self) -> int:
        """Current simulation timestep."""
        pass


class BaseEngine(ABC):
    """
    Abstract base class for physics engines.

    Engines manage global state (device, precision) and create scenes.
    """

    @abstractmethod
    def init(self, backend: str, precision: str) -> None:
        """
        Initialize the physics engine.

        Args:
            backend: Backend type - "gpu" or "cpu"
            precision: Numerical precision - "32" or "64"
        """
        pass

    @abstractmethod
    def create_scene(
        self,
        show_viewer: bool,
        sim_options: Dict,
        rigid_options: Dict,
        vis_options: Optional[Dict] = None,
        viewer_options: Optional[Dict] = None
    ) -> BaseScene:
        """
        Create a simulation scene.

        Args:
            show_viewer: Whether to show visualization window
            sim_options: Simulation options (dt, etc.)
            rigid_options: Rigid body options (solver, collision, etc.)
            vis_options: Visualization options
            viewer_options: Viewer window options

        Returns:
            Created scene instance
        """
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """PyTorch device for tensor operations."""
        pass

    @property
    @abstractmethod
    def tc_float(self) -> torch.dtype:
        """PyTorch tensor float dtype (precision)."""
        pass
