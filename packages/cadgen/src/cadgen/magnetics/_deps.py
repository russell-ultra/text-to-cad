"""Lazy accessors for the ``magnetics`` extra's dependencies.

cadgen has no numpy or scipy dependency and must not gain one unconditionally
(the ``snapshot`` extra and its playwright import in ``snapshot_core`` are the
precedent). Nothing in :mod:`cadgen.magnetics` imports magpylib, numpy, scipy
or plotly at module scope; each function that needs one calls the accessor
here, which imports on first use and turns ``ImportError`` into a
:class:`~cadgen.magnetics.types.MagneticsDependencyError` whose message names
the fix -- so ``cadgen magnetics --help`` works with the extra absent, and a
real run without it exits 1 with one line instead of a traceback.
"""

from __future__ import annotations

from typing import Any

from cadgen.magnetics.types import MagneticsDependencyError

__all__ = ["HINT", "magpylib", "numpy", "scipy_rotation", "plotly"]

HINT = 'pip install "cadgen[magnetics]"'


def _missing(name: str, exc: ImportError) -> MagneticsDependencyError:
    return MagneticsDependencyError(
        f"cadgen magnetics needs the Python package {name!r}, which is not installed. "
        f"Install the magnetics extra: {HINT}"
    )


def magpylib() -> Any:
    """The ``magpylib`` package (>= 5.2, < 6)."""
    try:
        import magpylib
    except ImportError as exc:
        raise _missing("magpylib", exc) from exc
    return magpylib


def numpy() -> Any:
    """The ``numpy`` package."""
    try:
        import numpy
    except ImportError as exc:
        raise _missing("numpy", exc) from exc
    return numpy


def scipy_rotation() -> Any:
    """``scipy.spatial.transform.Rotation`` -- the orientation type magpylib uses."""
    try:
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise _missing("scipy", exc) from exc
    return Rotation


def plotly() -> Any:
    """The ``plotly`` package (``plotly.graph_objects`` is ``plotly().graph_objects``)."""
    try:
        import plotly
        import plotly.graph_objects  # noqa: F401 - binds the submodule for attribute access
    except ImportError as exc:
        raise _missing("plotly", exc) from exc
    return plotly
