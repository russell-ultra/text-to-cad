"""Permanent-magnet mechanism analysis on cadgen models (``cadgen[magnetics]``).

The product is force, torque and potential energy along a cadgen-declared
kinematic mate: load a STEP whose leaves are named ``mag:<grade>:<direction>``,
sweep the mate, read off tendency, equilibria and their stability. Field slices
are the explanatory picture. magpylib is the physics engine; this package
writes no field math.

Public surface (SI throughout; model units convert in ``load_scene`` only)::

    load_scene(step_path, mate_name=None) -> MagScene      # scene.load
    sweep(scene, cfg) -> SweepResult                       # physics.sweep
    field(scene, plane_spec, grid, q, held=None) -> FieldSlice
    write_report(result, slice, out_dir, plotly="embed") -> Path

plus the data types and exceptions of :mod:`cadgen.magnetics.types`.

Import cost: the types are stdlib dataclasses and load eagerly (an ``except
SceneError`` clause must not trigger a physics import). The four functions are
re-exported lazily so ``import cadgen.magnetics`` never touches magpylib, numpy,
scipy, plotly or the CAD kernel -- cadgen must import with the extra absent, and
``cadgen magnetics --help`` must answer without it.
"""

from __future__ import annotations

from typing import Any

from cadgen.magnetics.types import (  # noqa: F401 - re-exports
    DOF_UNITS,
    SCHEMA,
    UNITS,
    Convergence,
    ConvergenceWarning,
    Equilibrium,
    FieldSlice,
    MagneticsDependencyError,
    MagneticsError,
    MagnetSpec,
    MagScene,
    MateError,
    MateSpec,
    Sample,
    SceneError,
    SweepConfig,
    SweepResult,
    Verdict,
    sweep_result_to_json,
)

__all__ = [
    "load_scene",
    "sweep",
    "field",
    "write_report",
    "SCHEMA",
    "UNITS",
    "DOF_UNITS",
    "MagnetSpec",
    "MateSpec",
    "MagScene",
    "SweepConfig",
    "Sample",
    "Equilibrium",
    "Verdict",
    "Convergence",
    "SweepResult",
    "FieldSlice",
    "MagneticsError",
    "MagneticsDependencyError",
    "SceneError",
    "MateError",
    "ConvergenceWarning",
    "sweep_result_to_json",
]

# name -> (submodule, attribute). Dotted strings, imported on first touch; never hoist.
_LAZY: dict[str, tuple[str, str]] = {
    "load_scene": ("cadgen.magnetics.scene", "load"),
    "sweep": ("cadgen.magnetics.physics", "sweep"),
    "field": ("cadgen.magnetics.physics", "field"),
    "write_report": ("cadgen.magnetics.report", "write_report"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attribute = target
    return getattr(importlib.import_module(module_name), attribute)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
