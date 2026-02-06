"""
Genesis physics engine implementation.

This module wraps the Genesis physics engine to conform to the BaseEngine interface.
"""

from typing import Optional, List, Dict, Tuple
import torch
import genesis as gs

from add_gym.engine.base_engine import (
    BaseEngine,
    BaseScene,
    BaseEntity,
    BaseJoint,
    BaseLink,
    BaseCamera
)


class GenesisCamera(BaseCamera):
    """Wrapper for Genesis camera."""

    def __init__(self, gs_camera):
        self._camera = gs_camera

    def follow_entity(self, entity: BaseEntity) -> None:
        # Unwrap to get Genesis entity
        gs_entity = entity._entity if isinstance(entity, GenesisEntity) else entity
        self._camera.follow_entity(gs_entity)

    def start_recording(self) -> None:
        self._camera.start_recording()

    def stop_recording(self, filename: str, fps: int = 30) -> None:
        self._camera.stop_recording(filename, fps=fps)

    def render(self) -> None:
        self._camera.render()

    @property
    def _in_recording(self) -> bool:
        """Check if camera is currently recording."""
        return self._camera._in_recording


class GenesisLink(BaseLink):
    """Wrapper for Genesis link."""

    def __init__(self, gs_link):
        self._link = gs_link

    @property
    def idx(self) -> int:
        return self._link.idx

    @property
    def idx_local(self) -> int:
        return self._link.idx_local

    @property
    def name(self) -> str:
        return self._link.name


class GenesisJoint(BaseJoint):
    """Wrapper for Genesis joint."""

    def __init__(self, gs_joint):
        self._joint = gs_joint

    @property
    def dofs_idx(self) -> List[int]:
        return self._joint.dofs_idx

    @property
    def dofs_idx_local(self) -> List[int]:
        return self._joint.dofs_idx_local

    @property
    def dofs_limit(self) -> List[Tuple[float, float]]:
        return self._joint.dofs_limit

    @property
    def name(self) -> str:
        return self._joint.name


class GenesisEntity(BaseEntity):
    """Wrapper for Genesis entity."""

    def __init__(self, gs_entity):
        self._entity = gs_entity

    # ========== Position and Orientation ==========

    def get_pos(self) -> torch.Tensor:
        return self._entity.get_pos()

    def set_pos(self, pos: torch.Tensor, envs_idx: Optional[torch.Tensor] = None) -> None:
        self._entity.set_pos(pos, envs_idx=envs_idx)

    def get_quat(self) -> torch.Tensor:
        return self._entity.get_quat()

    def set_quat(self, quat: torch.Tensor, envs_idx: Optional[torch.Tensor] = None) -> None:
        self._entity.set_quat(quat, envs_idx=envs_idx)

    # ========== Velocities ==========

    def get_vel(self) -> torch.Tensor:
        return self._entity.get_vel()

    def get_ang(self) -> torch.Tensor:
        return self._entity.get_ang()

    # ========== Degrees of Freedom ==========

    def get_dofs_position(self) -> torch.Tensor:
        return self._entity.get_dofs_position()

    def set_dofs_position(
        self,
        position: torch.Tensor,
        envs_idx: Optional[torch.Tensor] = None,
        dofs_idx_local: Optional[List[int]] = None
    ) -> None:
        self._entity.set_dofs_position(
            position=position,
            envs_idx=envs_idx,
            dofs_idx_local=dofs_idx_local
        )

    def get_dofs_velocity(self) -> torch.Tensor:
        return self._entity.get_dofs_velocity()

    def set_dofs_velocity(
        self,
        velocity: torch.Tensor,
        envs_idx: Optional[torch.Tensor] = None
    ) -> None:
        self._entity.set_dofs_velocity(velocity, envs_idx=envs_idx)

    def control_dofs_position(
        self,
        position: torch.Tensor,
        dofs_idx_local: Optional[List[int]] = None
    ) -> None:
        self._entity.control_dofs_position(position=position, dofs_idx_local=dofs_idx_local)

    def set_dofs_kp(self, kp: torch.Tensor) -> None:
        self._entity.set_dofs_kp(kp)

    def set_dofs_kv(self, kv: torch.Tensor) -> None:
        self._entity.set_dofs_kv(kv)

    def zero_all_dofs_velocity(self, envs_idx: Optional[torch.Tensor] = None) -> None:
        self._entity.zero_all_dofs_velocity(envs_idx=envs_idx)

    # ========== Links ==========

    def get_links_pos(self) -> torch.Tensor:
        return self._entity.get_links_pos()

    def get_links_quat(self) -> torch.Tensor:
        return self._entity.get_links_quat()

    def get_links_net_contact_force(self) -> torch.Tensor:
        return self._entity.get_links_net_contact_force()

    # ========== Contacts ==========

    def get_contacts(
        self,
        with_entity: BaseEntity,
        exclude_self_contact: bool = True
    ) -> Dict[str, torch.Tensor]:
        # Unwrap to get Genesis entity
        gs_entity = with_entity._entity if isinstance(with_entity, GenesisEntity) else with_entity
        return self._entity.get_contacts(
            with_entity=gs_entity,
            exclude_self_contact=exclude_self_contact
        )

    # ========== Geometry ==========

    def get_AABB(self) -> torch.Tensor:
        return self._entity.get_AABB()

    # ========== Structure Access ==========

    def get_joint(self, name: str) -> BaseJoint:
        gs_joint = self._entity.get_joint(name)
        return GenesisJoint(gs_joint)

    def get_link(self, name: str) -> BaseLink:
        gs_link = self._entity.get_link(name)
        return GenesisLink(gs_link)

    @property
    def joints(self) -> List[BaseJoint]:
        return [GenesisJoint(j) for j in self._entity.joints]

    @property
    def links(self) -> List[BaseLink]:
        return [GenesisLink(l) for l in self._entity.links]

    @property
    def n_dofs(self) -> int:
        return self._entity.n_dofs

    # ========== Special Methods ==========

    def set_qpos(self, qpos: torch.Tensor, envs_idx: Optional[torch.Tensor] = None) -> None:
        self._entity.set_qpos(qpos, envs_idx=envs_idx)


class GenesisScene(BaseScene):
    """Wrapper for Genesis scene."""

    def __init__(self, gs_scene):
        self._scene = gs_scene

    def add_entity(
        self,
        morph_type: str,
        morph_file: Optional[str] = None,
        morph_pos: Optional[Tuple[float, float, float]] = None,
        morph_quat: Optional[Tuple[float, float, float, float]] = None,
        material_type: str = "rigid",
        visualize_contact: bool = False
    ) -> BaseEntity:
        """Add entity to scene using Genesis API."""

        # Create morph based on type
        if morph_type == "plane":
            morph = gs.morphs.Plane()
        elif morph_type == "urdf":
            if morph_file is None:
                raise ValueError("morph_file required for URDF entity")
            morph = gs.morphs.URDF(
                file=morph_file,
                pos=morph_pos or (0.0, 0.0, 0.0),
                quat=morph_quat or (1.0, 0.0, 0.0, 0.0)
            )
        elif morph_type == "mjcf":
            if morph_file is None:
                raise ValueError("morph_file required for MJCF entity")
            morph = gs.morphs.MJCF(
                file=morph_file,
                pos=morph_pos or (0.0, 0.0, 0.0),
                quat=morph_quat or (1.0, 0.0, 0.0, 0.0)
            )
        else:
            raise ValueError(f"Unknown morph type: {morph_type}")

        # Create material
        if material_type == "rigid":
            material = gs.materials.Rigid()
        else:
            raise ValueError(f"Unknown material type: {material_type}")

        # Add entity to Genesis scene
        gs_entity = self._scene.add_entity(
            morph=morph,
            material=material,
            visualize_contact=visualize_contact
        )

        return GenesisEntity(gs_entity)

    def add_camera(
        self,
        res: Tuple[int, int],
        pos: Tuple[float, float, float],
        lookat: Tuple[float, float, float],
        fov: float
    ) -> BaseCamera:
        """Add camera to scene."""
        gs_camera = self._scene.add_camera(
            res=res,
            pos=pos,
            lookat=lookat,
            fov=fov
        )
        return GenesisCamera(gs_camera)

    def build(self, n_envs: int, env_spacing: Tuple[float, float]) -> None:
        """Build scene with parallel environments."""
        self._scene.build(n_envs=n_envs, env_spacing=env_spacing)

    def step(self) -> None:
        """Step simulation."""
        self._scene.step()

    @property
    def t(self) -> int:
        """Current simulation timestep."""
        return self._scene.t


class GenesisEngine(BaseEngine):
    """Genesis physics engine implementation."""

    def __init__(self, **kwargs):
        """Initialize Genesis engine.

        Args:
            **kwargs: Configuration parameters from Hydra (stored but not used by engine itself).
                     These are typically accessed directly from config by the environment.
        """
        self._gs = gs
        self._device = None
        self._tc_float = None
        self._config = kwargs  # Store config for potential future use

    def init(self, backend: str, precision: str) -> None:
        """Initialize Genesis engine."""
        # Map backend string to Genesis constants
        if backend == "gpu":
            gs_backend = gs.gpu
        elif backend == "cpu":
            gs_backend = gs.cpu
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Initialize Genesis
        gs.init(backend=gs_backend, precision=precision)

        # Store device and dtype for access
        self._device = gs.device
        self._tc_float = gs.tc_float

    def create_scene(
        self,
        show_viewer: bool,
        sim_options: Dict,
        rigid_options: Dict,
        vis_options: Optional[Dict] = None,
        viewer_options: Optional[Dict] = None
    ) -> BaseScene:
        """Create a Genesis scene."""

        # Convert options dicts to Genesis option objects
        gs_sim_options = gs.options.SimOptions(
            dt=sim_options.get("dt", 0.01)
        )

        # Build rigid options
        gs_rigid_options = gs.options.RigidOptions(
            dt=rigid_options.get("dt", 0.01),
            constraint_solver=getattr(gs.constraint_solver, rigid_options.get("constraint_solver", "Newton")),
            enable_collision=rigid_options.get("enable_collision", True),
            enable_self_collision=rigid_options.get("enable_self_collision", True),
            enable_joint_limit=rigid_options.get("enable_joint_limit", True)
        )

        # Build vis options if provided
        gs_vis_options = None
        if vis_options:
            gs_vis_options = gs.options.VisOptions(
                rendered_envs_idx=vis_options.get("rendered_envs_idx", [0]),
                show_world_frame=vis_options.get("show_world_frame", False)
            )

        # Build viewer options if provided
        gs_viewer_options = None
        if viewer_options:
            gs_viewer_options = gs.options.ViewerOptions(
                camera_pos=viewer_options.get("camera_pos", (3.5, 0.0, 2.5)),
                camera_lookat=viewer_options.get("camera_lookat", (0.0, 0.0, 0.5)),
                camera_fov=viewer_options.get("camera_fov", 40),
                max_FPS=viewer_options.get("max_FPS", 60)
            )

        # Create Genesis scene
        gs_scene = gs.Scene(
            show_viewer=show_viewer,
            sim_options=gs_sim_options,
            rigid_options=gs_rigid_options,
            vis_options=gs_vis_options,
            viewer_options=gs_viewer_options,
            profiling_options=gs.options.ProfilingOptions(show_FPS=False)
        )

        return GenesisScene(gs_scene)

    @property
    def device(self) -> torch.device:
        """PyTorch device for tensors."""
        if self._device is None:
            raise RuntimeError("Engine not initialized. Call init() first.")
        return self._device

    @property
    def tc_float(self) -> torch.dtype:
        """PyTorch tensor float dtype."""
        if self._tc_float is None:
            raise RuntimeError("Engine not initialized. Call init() first.")
        return self._tc_float
