"""Environment-variable expansion for config paths.

Configs are shared across machines, so paths are written against
``$ROBOREEL_DATA`` / ``$ROBOREEL_CKPT`` rather than hard-coded. The checkpoints
published on HuggingFace carry the same placeholders in their
``train_config.json``.

Kept dependency-free so the JAX, PyTorch and Robomimic entrypoints can all
import it without pulling in anything else.
"""

from __future__ import annotations

import os
import re

_VAR = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")


def expand_paths(value):
    """Expand ``${VAR}`` and ``~`` through a parsed config, recursively.

    An unset variable raises rather than leaving a literal ``${VAR}`` in the
    path, which would otherwise surface much later as a confusing
    ``FileNotFoundError``.
    """
    if isinstance(value, dict):
        return {k: expand_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_paths(v) for v in value]
    if not isinstance(value, str) or "$" not in value:
        return value
    expanded = os.path.expanduser(os.path.expandvars(value))
    if "$" in expanded:
        missing = sorted({m.group(1) for m in _VAR.finditer(expanded)})
        raise ValueError(
            f"config references undefined environment variable(s) {missing} in "
            f"{value!r} — export them first, e.g. `export "
            f"ROBOREEL_DATA=/path/to/dataset ROBOREEL_CKPT=/path/to/checkpoints`."
        )
    return expanded
