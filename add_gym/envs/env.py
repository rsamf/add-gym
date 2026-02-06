import torch
from hydra.utils import instantiate
from add_gym.robot import Manipulator
from pathlib import Path
from add_gym.learning.base_agent import AgentMode


class Environment:
    def __init__(self, config: dict, device: str):
        """
        Environment base class for Robotic Manipulation tasks.
        Contains the following components:
        - Engine (Genesis simulation)
        - Robot
        - Agent (Model + Learning Algorithm)

        Args:
            config: Environment configuration dictionary
        """
        self.device = device

        # configs
        self.env_cfg = config
        self.engine_cfg = config["engine"]
        self.robot_cfg = config["robot"]
        self.task_cfg = config["task"]
        self.ctrl_dt = self.engine_cfg["ctrl_dt"]

        if not self.engine_cfg["enable_viewer"]:
            import pyglet

            pyglet.options["headless"] = True

        # Initialize physics engine from Hydra config
        self.engine = instantiate(self.engine_cfg)

        # Initialize engine with appropriate backend
        if torch.cuda.is_available():
            self.engine.init(backend="gpu", precision="32")
        else:
            self.engine.init(backend="cpu", precision="32")

        # Build visualization and viewer options
        if self.engine_cfg["enable_viewer"]:
            vis_options = {
                "rendered_envs_idx": list(range(self.engine_cfg["num_envs"])),
                "show_world_frame": True,
            }
            viewer_options = {
                "camera_pos": (-3.5, -3.5, 3.5),
                "camera_lookat": (0.0, 0.0, 1.0),
                "camera_fov": 40,
                "max_FPS": 60,
            }
        else:
            vis_options = None
            viewer_options = None

        # Create scene through engine abstraction
        self.scene = self.engine.create_scene(
            show_viewer=self.engine_cfg["enable_viewer"],
            sim_options={"dt": self.ctrl_dt},
            rigid_options={
                "dt": self.ctrl_dt,
                "constraint_solver": "Newton",
                "enable_collision": True,
                "enable_self_collision": True,
                "enable_joint_limit": True,
            },
            vis_options=vis_options,
            viewer_options=viewer_options,
        )

        # Add ground plane
        self.plane = self.scene.add_entity(morph_type="plane")
        self.robot = Manipulator(
            num_envs=self.engine_cfg["num_envs"],
            scene=self.scene,
            engine=self.engine,  # Pass engine to robot
            robot_cfg=self.robot_cfg,
            env_spacing=self.engine_cfg["env_spacing"],
            enable_ref=self.task_cfg.get("visualize_ref_char", False),
            device=self.device,
        )
        self.enable_video_recording = self.engine_cfg.get(
            "enable_video_recording", True
        )

        if self.enable_video_recording:
            self.log_camera = self.scene.add_camera(
                res=(640, 480),
                pos=(-3.5, -3.5, 3.5),
                lookat=(0.0, 0.0, 1.0),
                fov=70,
            )
            self.video_length = self.engine_cfg.get("video_length", 20) * (
                1.0 / self.ctrl_dt
            )  # 20 seconds at 60 FPS
            self.record_start_time = 0
            self.log_dir = Path(self.env_cfg.get("log_dir", "logs")) / self.env_cfg.get(
                "experiment_name", "default_exp"
            )

        # Build the scene
        env_spacing = self.engine_cfg["env_spacing"]
        self.scene.build(
            n_envs=self.engine_cfg["num_envs"], env_spacing=(env_spacing, env_spacing)
        )
        self.robot.on_build()

        # VecEnv attributes
        self.num_envs = self.engine_cfg["num_envs"]

        # == init buffers ==
        self._init_buffers()

    def _init_buffers(self) -> None:
        self.time_buf = torch.zeros(
            self.num_envs, device=self.engine.device, dtype=torch.float32
        )

        self.extras = dict()

    def _video_step(self):
        if not self.enable_video_recording:
            return

        if not self.log_camera._in_recording:
            if self.scene.t % self.engine_cfg["video_interval"] == 0:
                print("Starting video recording...")
                self.record_start_time = self.scene.t
                self.log_camera.start_recording()
        else:
            if self.scene.t - self.record_start_time >= self.video_length:
                self.log_camera.stop_recording(
                    self.log_dir / f"video_{self.record_start_time}.mp4",
                    fps=round(1.0 / self.ctrl_dt),
                )
            else:
                self.log_camera.render()

    def set_mode(self, mode):
        if mode == AgentMode.TRAIN:
            self.num_envs = self.engine_cfg["num_envs"]
        elif mode == AgentMode.TEST:
            self.num_envs = 1
        else:
            raise ValueError("Unsupported agent mode: {}".format(mode))

    def step(self, actions: torch.Tensor) -> None:
        self._video_step()

        self.robot.apply_action(actions)
        self.scene.step()
        self.time_buf += self.ctrl_dt

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        if len(envs_idx) == 0:
            return

        self.time_buf[envs_idx] = 0
        # Reset robot
        # self.robot.reset(envs_idx)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            reset_env_ids = torch.arange(
                self.num_envs, device=self.device, dtype=torch.long
            )
        else:
            reset_env_ids = env_ids

        self.reset_idx(reset_env_ids)


class ImitationEnvironment(Environment):
    def __init__(self, config, device):
        super().__init__(config, device)
        self._diagnostics = {}

    def get_reward_succ(self):
        # setting the done flag flat to fail at the end of the motion avoids the
        # local minimal of a character just standing still until the end of the motion
        return 0.0

    def get_reward_fail(self):
        return 0.0

    def get_diagnostics(self):
        return self._diagnostics
