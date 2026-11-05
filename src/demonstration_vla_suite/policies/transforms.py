import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


#: Robot-side observation cameras the Pi0.5 path consumes (one ``image`` slot each;
#: cameras absent from a sample are zero-filled and masked out). Matches the layout
#: ``LfOVLADataset`` emits for ``robot_cameras`` (``observation/<cam>`` keys).
DEFAULT_ROBOT_CAMERAS = ("base_camera", "left_camera", "right_camera", "hand_camera")


def make_example() -> dict:
    """Creates a random input example for the demo VLA policy."""
    ex = {
        "observation/state": np.random.rand(8),
        "prompt": "do something",
    }
    for cam in DEFAULT_ROBOT_CAMERAS:
        ex[f"observation/{cam}"] = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
    return ex


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        assert np.abs(image).max() <= 1.0, "Image is not normalized"
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DemoVLAInputs(transforms.DataTransformFn):
    """Converts raw observations to model input format.

    One ``image`` slot is produced per entry in ``robot_cameras`` (in order). For each
    camera the image comes from ``data["observation/<cam>"]`` (mask ``True``); cameras
    absent from the sample are zero-filled with mask ``False`` so the model still sees a
    fixed-length image dict. ``observation/image`` is accepted as a legacy alias for the
    first camera. The Pi0 encoder iterates ``obs.images`` generically, so the slot names
    are arbitrary as long as training and eval agree.
    """

    model_type: _model.ModelType
    robot_cameras: tuple[str, ...] = DEFAULT_ROBOT_CAMERAS

    def __call__(self, data: dict) -> dict:
        cams = tuple(self.robot_cameras) or DEFAULT_ROBOT_CAMERAS
        present: dict[str, np.ndarray] = {}
        for cam in cams:
            key = f"observation/{cam}"
            if key in data:
                present[cam] = _parse_image(data[key])
        # Legacy alias: ``observation/image`` is the first robot camera.
        if cams[0] not in present and "observation/image" in data:
            present[cams[0]] = _parse_image(data["observation/image"])
        if not present:
            raise KeyError(
                f"DemoVLAInputs: sample has no robot camera image (looked for observation/<cam> "
                f"for cameras {cams} and the observation/image alias)."
            )
        ref = next(iter(present.values()))
        image = {cam: present.get(cam, np.zeros_like(ref)) for cam in cams}
        image_mask = {cam: np.bool_(cam in present) for cam in cams}

        inputs = {
            "state": data["observation/state"],
            "image": image,
            "image_mask": image_mask,
            # perceptual demonstration
            "static_image_emb": data.get("static_image_emb", None),
            "static_pos_emb": data.get("static_pos_emb", None),
            "static_state_emb": data.get("static_state_emb", None),
            "static_mask": data.get("static_mask", None),
            # recurrent demonstration
            "recur_image_emb": data.get("recur_image_emb", None),
            "recur_pos_emb": data.get("recur_pos_emb", None),
            "recur_state_emb": data.get("recur_state_emb", None),
            "recur_mask": data.get("recur_mask", None),
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DemoVLAOutputs(transforms.DataTransformFn):
    """Converts model outputs back to action space."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])}
