from __future__ import annotations

import base64
import json

import cv2
import numpy as np
from openai import OpenAI

_SYSTEM_PROMPT = (
    "You are a robot manipulation analysis assistant. "
    "Identify keyframes — frames that capture a distinct, task-relevant state "
    "transition: gripper opens or closes, contact with an object is established "
    "or released, an object starts or stops moving, or a sub-goal is reached. "
    "Return ONLY a JSON object with a single key \"keyframes\" whose value is a "
    "list of integer frame indices from those shown. Do not include any other text. "
    'Example: {"keyframes": [0, 42, 111]}'
)


def _encode_frame(frame: np.ndarray, jpeg_quality: int) -> str:
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), encode_params)
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def select_keyframes(
    frames: list[np.ndarray],
    frame_indices: list[int],
    task_description: str,
    model: str = "gpt-4o-mini",
    jpeg_quality: int = 85,
) -> list[int]:
    """Send frames to the VLM and return keyframe indices (original video frame numbers).

    Raises ValueError if the response cannot be parsed or contains invalid indices.
    """
    client = OpenAI()

    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"Task: {task_description}\n\n"
                f"Below are {len(frames)} frames sampled from the episode. "
                "Each frame is labeled with its index in the original video.\n"
            ),
        }
    ]

    for idx, frame in zip(frame_indices, frames):
        content.append({"type": "text", "text": f"[Frame {idx}]"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{_encode_frame(frame, jpeg_quality)}",
                    "detail": "low",
                },
            }
        )

    content.append(
        {"type": "text", "text": "Identify the keyframes for this task. Return the JSON object only."}
    )

    # Reasoning models (gpt-5 family, o1 / o3) burn tokens on internal
    # reasoning before producing output, so we need a much larger budget
    # than the JSON-payload size and we cap reasoning to "minimal" since
    # the keyframe task is selection, not chain-of-thought. JSON-mode
    # forces well-formed output. ``response_format`` + reasoning args are
    # silently accepted by gpt-4o-class non-reasoning models too.
    req_kwargs: dict = dict(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        max_completion_tokens=4096,
        response_format={"type": "json_object"},
    )
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        req_kwargs["reasoning_effort"] = "minimal"

    response = client.chat.completions.create(**req_kwargs)
    msg = response.choices[0].message
    raw = (msg.content or "").strip()
    if not raw:
        # Surface enough context to debug truncation vs. refusal vs. filter.
        finish = response.choices[0].finish_reason
        usage = getattr(response, "usage", None)
        raise ValueError(
            f"VLM returned empty content (finish_reason={finish!r}, usage={usage}, "
            f"model={model!r}). If finish_reason='length' the reasoning ate the "
            f"budget — raise max_completion_tokens or lower reasoning_effort."
        )
    try:
        parsed = json.loads(raw)
        keyframes = parsed["keyframes"]
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Could not parse VLM response: {raw!r}") from e

    # The VLM occasionally hallucinates plausible-looking but unsampled frame
    # indices (especially on long demos with 800+ frames where
    # it tends to pick round numbers like 832 / 896 that aren't in the
    # sampled set). Drop those rather than fail the whole demo — the
    # downstream pipeline always pads keyframe_indices with the first and
    # last sampled frame anyway, so even an all-invalid response still
    # produces a usable 2-entry annotation. We log a warning so wholesale
    # invalid responses don't go unnoticed.
    import warnings
    valid = set(frame_indices)
    kept = [int(k) for k in keyframes if int(k) in valid]
    dropped = [int(k) for k in keyframes if int(k) not in valid]
    if dropped:
        warnings.warn(
            f"VLM returned {len(dropped)} index/indices not in the sampled "
            f"set (dropped): {dropped}. Kept {len(kept)} valid keyframe(s).",
            RuntimeWarning, stacklevel=2,
        )
    return kept
