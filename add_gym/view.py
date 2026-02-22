import hydra
import torch
import platform
import os.path
from omegaconf import DictConfig
from hydra.utils import instantiate
from add_gym.anim import motion_lib, kin_char_model
from add_gym.engine.base_engine import BaseEngine, BaseScene
import rerun as rr


class ViewManipulator:
    def __init__(
        self,
        scene: BaseScene,
        robot_cfg: dict,
        device: str = "cpu",
    ):
        self._device = device
        self._scene = scene
        self._args = robot_cfg

        # Build Kinematic Character Model (Required for MotionLib)
        self._build_kin_char_model(robot_cfg["urdf_path"])

        # Prepare Reference Entity (Visual Only)
        morph_path = robot_cfg["urdf_path"]
        morph_type = "urdf" if morph_path.endswith(".urdf") else "mjcf"

        self.ref_entity = scene.add_entity(
            morph_type=morph_type,
            morph_file=morph_path,
            morph_pos=(0.0, 0.0, 0.0),
            morph_quat=(1.0, 0.0, 0.0, 0.0),
            material_type="rigid",
            visualize_contact=True,
        )

        # Joint limits will be initialized after scene is built
        self.joint_limits = None
        self.non_root_joint_limits = None
        self.joint_names = []
        self.non_root_joint_names = []
        self._device = device

    def _initialize_joint_limits(self):
        """Initialize joint limits after scene is built."""
        # Build joint limits for normalization
        joint_limits_list = []
        joint_names_list = []
        for joint in self.ref_entity.joints:
            joint_limits_list.extend(joint.dofs_limit)
            # Store joint name for each DOF
            for _ in joint.dofs_limit:
                joint_names_list.append(joint.name)

        self.joint_limits = torch.tensor(
            joint_limits_list, device=self._device, dtype=torch.float32
        )  # (num_dofs, 2)
        # Extract just the non-root joint limits (skip first 6 for floating base)
        self.non_root_joint_limits = self.joint_limits[6:]
        self.joint_names = joint_names_list
        self.non_root_joint_names = joint_names_list[6:]

    def _build_kin_char_model(self, char_file):
        _, file_ext = os.path.splitext(char_file)
        if file_ext == ".xml":
            char_model = kin_char_model.KinCharModel(self._device)
        else:
            print("Unsupported character file format: {:s}".format(file_ext))
            assert False

        self._kin_char_model = char_model
        self._kin_char_model.load_char_file(char_file)

    # -- Properties mimicking Manipulator but using ref_entity --
    @property
    def base_pos(self):
        return self.ref_entity.get_pos()

    @property
    def base_quat(self):
        return self.ref_entity.get_quat()

    @property
    def base_lin_vel(self):
        return self.ref_entity.get_vel()

    @property
    def base_ang_vel(self):
        return self.ref_entity.get_ang()

    @property
    def link_pos(self):
        return self.ref_entity.get_links_pos()

    @property
    def link_quat(self):
        return self.ref_entity.get_links_quat()

    @property
    def dof_pos(self):
        # Match Manipulator behavior: slicing to exclude root if standard
        # However, check if ref_entity actually has root dofs in dof_pos?
        # Genesis usually puts 6 root dofs at start for floating base.
        # Ideally we check self.joint_lookup["base"] but slicing 6: is standard for Manipulator.
        return self.ref_entity.get_dofs_position()[:, 6:]

    @property
    def dof_vel(self):
        return self.ref_entity.get_dofs_velocity()[:, 6:]


class ViewEnvironment:
    def __init__(self, config: dict, device: str):
        self.device = device
        self.env_cfg = config
        self.engine_cfg = config["engine"]
        self.robot_cfg = config["robot"]
        self.task_cfg = config["task"]
        self.ctrl_dt = self.engine_cfg["ctrl_dt"]

        # Initialize physics engine from Hydra config
        self.engine = instantiate(self.engine_cfg)

        # Initialize engine with appropriate backend
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            self.engine.init(backend="gpu", precision="32")
        else:
            self.engine.init(backend="cpu", precision="32")

        # Create Scene
        self.scene = self.engine.create_scene(
            show_viewer=True,
            sim_options={"dt": self.ctrl_dt},
            rigid_options={
                "dt": self.ctrl_dt,
                "constraint_solver": "Newton",
                "enable_collision": False,
            },
            vis_options={
                "rendered_envs_idx": list(range(self.engine_cfg["num_envs"])),
                "show_world_frame": True,
            },
            viewer_options={
                "camera_pos": (-3.5, -3.5, 3.5),
                "camera_lookat": (0.0, 0.0, 1.0),
                "camera_fov": 40,
            },
        )
        self.plane = self.scene.add_entity(morph_type="plane")

        # Create Robot (ViewManipulator)
        self.robot = ViewManipulator(
            scene=self.scene,
            robot_cfg=self.robot_cfg,
            device=self.device,
        )

        # Video Recording Setup
        self.video_output_dir = self.env_cfg.get("video_output_dir")
        if self.video_output_dir:
            import pathlib

            self.video_output_dir = pathlib.Path(self.video_output_dir)
            self.video_output_dir.mkdir(parents=True, exist_ok=True)

            self.log_camera = self.scene.add_camera(
                res=(1280, 720),
                pos=(2.5, 2, 2),
                lookat=(0.0, 0.0, 1.0),
                fov=60,
            )
            self.is_recording = True

        self._init_buffers()

        # Build scene
        self.scene.build(
            n_envs=1,
            env_spacing=(0.,0.),
        )

        # Initialize joint limits after scene is built
        self.robot._initialize_joint_limits()

        if self.video_output_dir:
            self.log_camera.follow_entity(self.robot.ref_entity)
            self.log_camera.start_recording()

        # Motion Library Initialization
        self._kin_char_model = self.robot._kin_char_model
        self.motion_lib = motion_lib.MotionLib(
            motion_file=self.task_cfg["motion_file"],
            motion_order=list(self.task_cfg["motion_joint_order"]),
            kin_char_model=self._kin_char_model,
            dt=self.ctrl_dt,
            device=device,
        )

        # Preallocate tensors for reference motion
        # We can use the properties from ViewManipulator (which point to ref_entity)
        # Note: ref_entity values after build might be zero or default pose, which is fine for init.
        root_pos = self.robot.base_pos
        root_rot = self.robot.base_quat
        root_vel = self.robot.base_lin_vel
        root_ang_vel = self.robot.base_ang_vel
        dof_pos = self.robot.dof_pos
        dof_vel = self.robot.dof_vel

        self.ref_root_pos = torch.zeros_like(root_pos)
        self.ref_root_rot = torch.zeros_like(root_rot)
        self.ref_root_vel = torch.zeros_like(root_vel)
        self.ref_root_ang_vel = torch.zeros_like(root_ang_vel)
        self.ref_dof_pos = torch.zeros_like(dof_pos)
        self.ref_dof_vel = torch.zeros_like(dof_vel)

    def _init_buffers(self):
        self.time_buf = torch.zeros(1, device=self.engine.device, dtype=torch.float32)

    def reset(self, env_ids=None):
        if env_ids is None:
            reset_env_ids = torch.arange(1, device=self.engine.device, dtype=torch.long)
        else:
            reset_env_ids = env_ids

        self.time_buf[reset_env_ids] = 0

        self.update_ref_motion()  # Initial update
        self.reset_ref_char()

    def reset_ref_char(self):
        qpos = torch.cat(
            [
                self.ref_root_pos,
                self.ref_root_rot,
                self.ref_dof_pos,
            ],
            dim=-1,
        )
        self.robot.ref_entity.set_qpos(qpos)
        velocity = torch.cat(
            [
                self.ref_root_vel,
                self.ref_root_ang_vel,
                self.ref_dof_vel,
            ],
            dim=-1,
        )
        self.robot.ref_entity.set_dofs_velocity(velocity)

    def update_ref_motion(self):
        # Calculate current motion time
        motion_times = self.time_buf

        # Handle Looping
        motion_lengths = self.motion_lib.get_motion_length(0)
        motion_times = motion_times % motion_lengths

        # Query Motion Lib
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel = (
            self.motion_lib.get_precomputed_motion_step(0, motion_times)
        )

        # Update internal reference state buffers
        self.ref_root_pos[:] = root_pos
        self.ref_root_rot[:] = root_rot
        self.ref_root_vel[:] = root_vel
        self.ref_root_ang_vel[:] = root_ang_vel
        self.ref_dof_vel[:] = dof_vel
        self.ref_dof_pos[:] = dof_pos

        # Update Visual Reference Character
        qpos = torch.cat(
            [
                self.ref_root_pos,
                self.ref_root_rot,
                self.ref_dof_pos,
            ],
            dim=-1,
        )
        self.robot.ref_entity.set_qpos(qpos)
        velocity = torch.cat(
            [
                self.ref_root_vel,
                self.ref_root_ang_vel,
                self.ref_dof_vel,
            ],
            dim=-1,
        )
        self.robot.ref_entity.set_dofs_velocity(velocity)

    def _video_step(self):
        if not self.video_output_dir:
            return

        if self.is_recording:
            self.log_camera.render()

    def _log_joint_rotations(self):
        """Log normalized joint rotations to rerun."""
        # Set the current simulation time
        rr.set_time("sim_time", duration=self.time_buf[0].item())

        # Get current joint positions (already excludes root)
        dof_pos = self.robot.dof_pos[0]  # Take first env

        # Get joint limits
        joint_limits = self.robot.non_root_joint_limits
        joint_names = self.robot.non_root_joint_names

        # Normalize to [-1, 1] range
        lower_limits = joint_limits[:, 0]
        upper_limits = joint_limits[:, 1]

        # Normalize: (pos - mid) / (range/2) = (2*pos - (upper+lower)) / (upper-lower)
        mid_point = (upper_limits + lower_limits) / 2.0
        half_range = (upper_limits - lower_limits) / 2.0
        normalized_pos = (dof_pos - mid_point) / half_range

        # Clamp to [-1, 1] to handle any numerical issues
        normalized_pos = torch.clamp(normalized_pos, -1.0, 1.0)

        # Log each joint rotation
        for name, norm_val in zip(joint_names, normalized_pos):
            rr.log(
                f"joints/{name}",
                rr.Scalars(norm_val.item())
            )

    def save_video(self):
        if self.video_output_dir and self.is_recording:
            import time

            filename = self.video_output_dir / f"video_{int(time.time())}.mp4"
            print(f"Stopping video recording. Saving to {filename}...")
            self.log_camera.stop_recording(str(filename), fps=round(1 / self.ctrl_dt))
            self.is_recording = False

    def step(self):
        # Update motion logic using current time
        self.update_ref_motion()

        # Log joint rotations to rerun
        self._log_joint_rotations()

        # Handle video recording (camera updates and rendering)
        self._video_step()

        # Step scene (visualizer update)
        self.scene.step()
        self.time_buf += self.ctrl_dt


@hydra.main(version_base=None, config_path="configs", config_name="view")
def main(cfg: DictConfig):
    # Ensure visualization is enabled
    if "task" not in cfg:
        print("Error: 'task' config not found.")
        return

    cfg.task.visualize_ref_char = True

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Initialize rerun for logging
    rr.init("g1_joint_visualization", spawn=True)

    # Initialize Custom View Environment
    env = ViewEnvironment(cfg, device=device)

    print("Starting visualization loop...")
    print("Press Ctrl+C to stop.")

    env.reset()

    try:
        while True:
            # Step execution
            env.step()

    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")
        env.save_video()


if __name__ == "__main__":
    main()
