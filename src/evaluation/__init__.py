"""Evaluation package: shared utilities + per-model runners + the protocol driver.

``scripts/evaluation.py`` is the single entry point; it picks a runner from the eval
config's ``model`` field and hands it to :func:`evaluation.utils.run_eval_protocol`.
"""
