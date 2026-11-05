"""numpy 2.x -> 1.x unpickling shim.

Stage 0a (``process_training_robot_trajectories.py``) runs in the ``uniskill`` conda
env, which has numpy 2.x, so the per-sample ``postprocessed_robot_trajectories``
pkls are pickled under numpy 2.x. numpy 2.x renamed its internal ``numpy.core``
package to ``numpy._core``; a numpy 1.x interpreter (the ``pi05``
env) can't resolve ``numpy._core.*`` when unpickling those arrays and raises
``ModuleNotFoundError: No module named 'numpy._core...'``.

Importing this module aliases ``numpy._core`` (and its submodules) onto the
still-present ``numpy.core`` so ``pickle.load`` resolves the unchanged array
classes. It is a no-op under numpy 2.x.
"""
import sys

import numpy

if not hasattr(numpy, "_core"):
    import numpy.core as _np_core

    sys.modules["numpy._core"] = _np_core
    for _sub in ("numeric", "multiarray", "umath", "_multiarray_umath",
                 "_exceptions", "_dtype", "_dtype_ctypes", "fromnumeric",
                 "_methods", "overrides"):
        try:
            sys.modules["numpy._core." + _sub] = __import__(
                "numpy.core." + _sub, fromlist=[_sub])
        except Exception:
            pass
