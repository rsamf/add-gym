import enum
import pickle
import numpy as np


class LoopMode(enum.Enum):
    CLAMP = 0
    WRAP = 1


def convert_to_pkl_motion(
    input_file: str,
    output_file: str,
    loop_mode: LoopMode = LoopMode.CLAMP,
    fps: int = 30,
):
    """
    Convert a motion file to pickle format.

    Args:
        input_file: Path to input motion file
        output_file: Path to output pickle file
        loop_mode: Loop mode (CLAMP or WRAP)
        fps: Frames per second
    """
    frames = []
    with open(input_file, "r") as in_f:
        for line in in_f:
            frame_data = [float(val) for val in line.strip().split(",")]
            frames.append(frame_data)

    frames_array = np.array(frames)

    motion_data = Motion(loop_mode=loop_mode, fps=fps, frames=frames_array)

    motion_data.save(output_file)
    return motion_data


def load_motion(file: str):
    if file.endswith(".motion"):
        return convert_to_pkl_motion(file, file.replace(".motion", ".pkl"))

    with open(file, "rb") as filestream:
        in_dict = pickle.load(filestream)

        loop_mode_val = in_dict["loop_mode"]
        fps = in_dict["fps"]
        frames = in_dict["frames"]

        loop_mode = LoopMode(loop_mode_val)

        motion_data = Motion(loop_mode=loop_mode, fps=fps, frames=frames)
    return motion_data


class Motion:
    def __init__(self, loop_mode, fps, frames):
        self.loop_mode = loop_mode
        self.fps = fps
        self.frames = frames

    def save(self, out_file):
        with open(out_file, "wb") as out_f:
            out_dict = {
                "loop_mode": self.loop_mode.value,
                "fps": self.fps,
                "frames": self.frames,
            }
            pickle.dump(out_dict, out_f)

    def get_length(self):
        num_frames = self.frames.shape[0]
        motion_len = float(num_frames - 1) / self.fps
        return motion_len
