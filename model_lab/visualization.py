"""Compatibility wrapper for the British-English :mod:`model_lab.visualisation` module.

New code should import from ``model_lab.visualisation``.  This wrapper is retained so that
existing local code using the earlier American-English module name continues to work.
"""

from .visualisation import *  # noqa: F401,F403
