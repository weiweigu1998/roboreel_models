"""Frozen SigLIP text tower used by the VTCL loss (Paper §II-E (4)).

Returns one ``(text_embed_dim,) np.float32`` embedding per task name. The
encoder is loaded once, run once over the unique task strings, and discarded
— the dataset caches the resulting array per task so the heavy model is not
held in memory after init.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Make src/lfo_task_descriptions.py importable from this submodule.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))  # .../action_understanding_bench/src
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from lfo_task_descriptions import load_gt_annotations  # noqa: E402


def build_text_embeddings(tasks: list[str], device: str = "cpu", *,
                          model_id: str = "google/siglip-base-patch16-224",
                          ) -> dict[str, np.ndarray]:
    """``{task: (D,) np.float32}`` via the frozen SigLIP text tower.

    One entry per ``tasks`` snake-case task name, keyed by that name
    (e.g. ``"pick_cube"``), with the curated GT sentence as the input text.

    Run once in the main process during dataset ``__init__``; the model is
    not kept around (we only need the embeddings).
    """
    import torch
    from transformers import AutoModel, AutoProcessor

    gt = load_gt_annotations()

    keys: list[str] = []
    sentences: list[str] = []
    for t in tasks:
        keys.append(t)
        sentences.append(gt[t])

    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    with torch.no_grad():
        inputs = proc(text=sentences, return_tensors="pt", padding=True).to(device)
        text_embeds = model.get_text_features(**inputs)  # (N, D)
    out = {k: text_embeds[i].detach().cpu().float().numpy()
           for i, k in enumerate(keys)}
    del model, proc, text_embeds
    return out
