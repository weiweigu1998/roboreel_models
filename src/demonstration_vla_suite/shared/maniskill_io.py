"""Shared observation resize, robot state packing, and live-env → policy-input adapters.

The ``extract_obs_for_pi05`` adapter reproduces
``LfOVLADataset.__getitem__``'s per-timestep output byte-for-byte (same 224² INTER_LINEAR
resize, same 8-D state = 7 joints + mean gripper, same per-camera key layout) so the
eval runners feed the policies exactly what training did.
"""

from __future__ import annotations

import logging

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# Must match eval policy buffer and Pi0.5 image path (then resize_with_pad is a no-op on 224²).


def resize_image(img: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Resize image to target_size x target_size (INTER_LINEAR). Same as eval_maniskill_vla."""
    import cv2

    if img.shape[0] != target_size or img.shape[1] != target_size:
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return img


def pack_robot_state_8d(
    joint_states_7: np.ndarray,
    gripper_fingers: np.ndarray,
) -> np.ndarray:
    """8-dim state: 7 arm joints + scalar gripper = mean(fingers).

    Matches ``extract_obs_from_maniskill`` in eval (``qpos[:7]`` and ``qpos[7:9].mean()``).
    ``gripper_fingers`` can be shape (2,) or any length; the mean is one scalar broadcast to dim 8.
    """
    joints = np.asarray(joint_states_7, dtype=np.float32).reshape(7)
    g = np.asarray(gripper_fingers, dtype=np.float32).reshape(-1)
    gripper_scalar = np.array([float(np.mean(g))], dtype=np.float32)
    return np.concatenate([joints, gripper_scalar])


def load_demo_frames_from_h5(h5_path: str, camera_key: str = "base_camera") -> np.ndarray:
    """Load RGB frames from ManiSkill trajectory HDF5. Returns uint8 [T, H, W, 3]."""
    with h5py.File(h5_path, "r") as f:
        rgb = f[f"traj_0/obs/sensor_data/{camera_key}/rgb"][()]
    logger.info("  Loaded %s frames from %s (shape: %s)", rgb.shape[0], h5_path, rgb.shape)
    return rgb


def load_demo_frames_from_video(video_path: str) -> np.ndarray:
    """Load RGB frames from mp4. Returns uint8 [T, H, W, 3]."""
    import imageio

    reader = imageio.get_reader(video_path, format="ffmpeg")
    frames = np.stack([frame for frame in reader])
    reader.close()
    logger.info("  Loaded %s frames from %s (shape: %s)", frames.shape[0], video_path, frames.shape)
    return frames


def prepare_demo_for_buffer(demo_frames: np.ndarray) -> np.ndarray:
    """Resize demo frames to 224x224 and add view dimension.

    Args:
        demo_frames: (T, H, W, 3) uint8

    Returns:
        (T, 1, 224, 224, 3) uint8 for ``DemoVLAPolicy.set_demonstration`` / fixed-demo encoding.
    """
    resized = np.stack([resize_image(f, 224) for f in demo_frames])
    return resized[:, np.newaxis, ...]


def demo_buffer_arrays_from_frames(demo_frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``images`` / ``state`` payload for ``DemoVLAPolicy.set_demonstration`` (expert demo).

    Uses zero states for demonstration frames (policy normalizes state embeddings).
    """
    images = prepare_demo_for_buffer(demo_frames)
    n = int(demo_frames.shape[0])
    states = np.zeros((n, 8), dtype=np.float32)
    return images, states


# ---------------------------------------------------------------------------
# Live-env → policy-input adapters. THESE MUST MATCH ``LfOVLADataset`` BYTE-FOR-BYTE.
# ---------------------------------------------------------------------------

def _extract_rgb_uint8_hwc(maniskill_obs: dict, camera: str) -> np.ndarray:
    """Pull RGB ``(H, W, 3)`` uint8 from a single-env ManiSkill obs.

    ManiSkill stores camera color under ``obs["sensor_data"][<camera>]["Color"]``
    as ``(B, H, W, 4)`` (RGBA) tensors when ``obs_mode="sensor_data"``. Some
    older replays use the key ``"rgb"``. We always strip alpha and the batch
    dim and return uint8.
    """
    cam = maniskill_obs["sensor_data"][camera]
    img = cam.get("Color", cam.get("rgb"))
    if img is None:
        raise KeyError(
            f"Camera {camera!r} has no 'Color'/'rgb' key. Available: {sorted(cam.keys())}"
        )
    # Lazy torch import — keeps this module usable in JAX-only envs.
    try:
        import torch as _torch  # noqa: WPS433
        if isinstance(img, _torch.Tensor):
            img = img.cpu().numpy()
    except ImportError:
        pass
    if img.ndim == 4:
        img = img[0]
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    return img


def _extract_qpos(maniskill_obs: dict) -> np.ndarray:
    """Pull the robot ``qpos`` vector from a single-env ManiSkill obs."""
    qpos = maniskill_obs["agent"]["qpos"]
    try:
        import torch as _torch  # noqa: WPS433
        if isinstance(qpos, _torch.Tensor):
            qpos = qpos.cpu().numpy()
    except ImportError:
        pass
    if qpos.ndim == 2:
        qpos = qpos[0]
    return np.asarray(qpos, dtype=np.float32)


def _normalize_camera_list(camera: str, cameras) -> list[str]:
    if cameras is None:
        return [camera]
    if isinstance(cameras, str):
        return [cameras]
    out = list(cameras)
    if not out:
        raise ValueError("cameras must contain at least one camera name.")
    return out


def extract_obs_for_pi05(
    maniskill_obs: dict,
    camera: str = "base_camera",
    *,
    cameras=None,
) -> dict:
    """Replicate ``LfOVLADataset.__getitem__``'s Pi0.5-side output (the raw dict that
    feeds ``DemoVLAInputs``).

    Returned keys::

        observation/<cam>  : (224, 224, 3) uint8   (HWC uint8 — one per camera)
        observation/image  : alias for the first camera
        observation/state  : (8,) float32

    ``cameras`` (a list) selects the camera set; ``camera`` is the single-camera
    fallback. Same ``resize_image`` (INTER_LINEAR) and ``pack_robot_state_8d`` calls
    the dataset class uses; kept uint8 because OpenPI's ``ResizeImages`` transform
    (a no-op here, since these are already 224²) takes HWC uint8.
    """
    cams = _normalize_camera_list(camera, cameras)
    out: dict = {}
    for cam in cams:
        out[f"observation/{cam}"] = resize_image(_extract_rgb_uint8_hwc(maniskill_obs, cam), 224)
    out["observation/image"] = out[f"observation/{cams[0]}"]
    qpos = _extract_qpos(maniskill_obs)
    out["observation/state"] = pack_robot_state_8d(qpos[:7], qpos[7:9])
    return out
