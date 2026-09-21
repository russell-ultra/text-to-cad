# cadgen.magnetics

Permanent-magnet mechanism analysis, behind the `magnetics` extra
(`pip install "cadgen[magnetics]"`). The product is force, torque and potential
energy along a cadgen-declared kinematic mate: load a STEP whose leaves are
named `mag:<grade>:<direction>`, sweep the mate, and read off the tendency, the
equilibria and their stability. Field slices are the explanatory picture, not
the answer. magpylib is the physics engine; this module writes no field math.

**PURPOSE** — the `cadgen magnetics inspect | sweep | field` verbs and the
library behind them: the name grammar and solid classification that turn a
magnet-bearing STEP into magpylib sources (`scene`), the mate algebra that poses
them (`kinematics`), the meshed force/torque, energy, equilibria, clearance and
convergence protocol that produce a verdict (`physics`), and the plotly report
and `sweep.json` they are written to (`report`).

**MAY DEPEND ON** — magpylib (the field/force engine), numpy, scipy and plotly,
each reached only through `_deps`; and cadgen's own scene reading, kinematics
sidecar and geometry queries, imported inside functions. Never at module scope.

**DEPENDED ON BY** — the `cadgen magnetics` command group and nothing else in
cadgen. The rest of the distribution imports and runs with the extra absent.

## The rest of the documentation

This file holds the LAWS of the module: statements a change can violate even
when it passes. Each carries a pressure-test to apply before writing code. The
mechanism each law constrains is specified where it lives — the module's own
source, `types.py` first (the frozen SI contract every other file codes
against). Where this file and a docstring disagree, the docstring is right and
this one is stale.

## The design laws

### 1. The heavy dependencies are gated and lazy

`magpylib`, `numpy`, `scipy` and `plotly` reach this module only through the
`_deps` accessors, which import on first use and turn `ImportError` into a
`MagneticsDependencyError` whose message names the fix. Nothing here — not
`types`, not the package `__init__`, not the CLI verb module — imports them, or
the CAD kernel, at module scope. `types` is stdlib-only and loads eagerly (an
`except SceneError` clause must not drag in a physics import); the four public
functions are re-exported lazily.

*Pressure-test*: with magpylib, numpy, scipy and plotly all unimportable,
`import cadgen.magnetics` succeeds and `cadgen magnetics --help` exits 0. A real
run without the extra exits 1 with the one-line `pip install "cadgen[magnetics]"`
hint, never a traceback.

### 2. The SI boundary is `scene.load`

Every value inside the module is SI: metres, radians, tesla, newtons,
newton-metres, joules. Model units — millimetres and degrees — enter at the
edges only: `scene.load` converts the geometry and the mate, and the CLI
converts `--at` and `--hold` into the `SweepConfig` it hands `physics.sweep`. No
function in `kinematics`, `physics` or `report` ever sees a millimetre or a
degree; `MagScene`, `MateSpec` and everything reachable from them are SI by
construction.

*Pressure-test*: a 6.35 mm cube through `scene.load` yields
`dimension == (0.00635,)*3` and the mate origin/limits in metres or radians. No
`* 1e-3` or degree→radian factor appears outside `scene.load` and the CLI's
argument conversion.

### 3. The body-frame rule — never rotate twice

magpylib's `polarization` and `dimension` are **body-frame**: the source rotates
them by its own `orientation`. So a classified cuboid or cylinder gets
`orientation = R_box` (the frame built from its world face normals), `dimension`
measured in that frame — not the world AABB — and `polarization = R_boxᵀ ·
J_world`. A meshed solid gets identity `orientation`, world-frame vertices, and
`polarization = J_world`. A value already carried by `orientation` is never
multiplied by the occurrence rotation a second time.

*Pressure-test*: the on-axis field of a 45°-tilted cube equals that of an
untilted control rotated by the same `R` (no double rotation), and its
`dimension` matches the body edge lengths, not the enlarged bounding box.

### 4. The weld rule — a mesh magnet is closed or it is an error

A solid that is neither a clean cuboid nor a clean cylinder becomes a
`TriangularMesh` from its tessellation, but build123d tessellation is unwelded
(a unit box tessellates to 24 per-face vertices, not 8). `weld_mesh` rounds
vertices to `WELD_TOL_M`, uniquifies, remaps faces and drops degenerate
triangles before the mesh is constructed with `check_open`,
`check_disconnected` and `reorient_faces` on. A leaf still open or disconnected
after welding is a `SceneError` naming the leaf and the failing check — never a
silently unreliable magnet.

*Pressure-test*: welding a tessellated box yields 8 vertices and 12 faces; a
deliberately open mesh raises `SceneError` naming the leaf.

### 5. `squeeze=False`, always — reduce shapes explicitly

Every `getFT` call passes `squeeze=False`, so the result is `(p, t, 3)` for
`p` poses and `t` targets regardless of how many there are, and the pivot is
`(t, p, 3)`. The group force and torque are `F.sum(axis=1)` and `T.sum(axis=1)`
→ `(p, 3)`. The module never relies on magpylib's squeezed shape, which changes
with the source and pose counts.

*Pressure-test*: 1×1, 1×N and M×N source/pose combinations all reduce to
`(p, 3)`. `getFT`'s signature is pinned with `inspect.signature`, so an upstream
change to its defaults or parameters fails a test loudly rather than silently
altering a shape.

### 6. Single-pose copies for grid refinement

The one vectorised sweep uses pathed sources (`pathed_sources`, position and
orientation paths on copies). Everything that refines the pose grid —
equilibrium bisection, `--relax` root-finding, the convergence protocol's nested
grid — evaluates on fresh single-pose copies (`posed_sources`), never on a
shared pathed source. A path source there would broadcast to
`(grid, targets, 3)` and couple the poses to each other.

*Pressure-test*: refine an equilibrium or relax a held DOF and confirm the
coarse sweep's arrays are unchanged; a grid axis produces one force per pose,
not a cross-product over the sweep's path.

### 7. Every verdict carries convergence — indeterminate beats wrong

An answer is trustworthy only with an uncertainty, and the uncertainty comes
from refining three axes **separately**: the mesh at fixed `eps`, `eps` at the
finest mesh, and a nested `N → 2N−1` pose grid. Each relative delta is guarded
by `Q_scale = converge_tol · max|Q|` so a metric near `Q ≈ 0` never divides by
zero. `converged` is `True` only when every axis is within `converge_tol` and
the equilibria and energy agree; `False` records the axes at fault in `reasons`;
`None` for `--convergence off`. A non-converged run still exits 0 and emits a
`ConvergenceWarning` — the analysis ran, its answer is just not trusted.

*Pressure-test*: `off`, `quick` and `full` run the documented level counts; the
band widens where refinement moves the answer; a tendency inside its own
uncertainty is reported `indeterminate`, not guessed.

### 8. The verdict is split — three claims, never collapsed

A `Verdict` states three independent things and never conflates them:

- `tendency` — `"+axis"`, `"-axis"`, or `"indeterminate"` when `|Q| <
  uncertainty`. Which way the DOF is magnetically pushed at the verdict pose.
- `exceeds_threshold` — `|Q|` against the friction threshold; `None` when no
  threshold was given or the band `[|Q| − u, |Q| + u]` straddles it.
- `admissible` — `False` when the tendency points outside the mate limits at a
  limit pose, or into an interval clearance marked inaccessible.

Rest under friction is a **band** around each stable equilibrium, not a point.
Transverse load `|F⊥|` and torque are guide loads reported for the mechanism;
they are never fed back in as friction inputs.

*Pressure-test*: at a lower limit with a `-axis` tendency, `admissible` is
`False`; with `|Q| < uncertainty`, `tendency` is `indeterminate` and
`exceeds_threshold` is `None`; with no `--friction-*`, `exceeds_threshold` is
`None` however large `|Q|` is.

## The shape of the module

```
cadgen/magnetics/
  types.py        # the frozen SI contract: MagnetSpec, MateSpec, MagScene,
                  #   SweepConfig, Sample, Equilibrium, Verdict, Convergence,
                  #   SweepResult, FieldSlice, the exception tree, and
                  #   sweep_result_to_json (the magnetics-report/1 serializer)
  _deps.py        # magpylib(), numpy(), scipy_rotation(), plotly() -- lazy,
                  #   each raising MagneticsDependencyError with the pip hint
  __init__.py     # eager type re-exports; load_scene/sweep/field/write_report
                  #   re-exported lazily via __getattr__
  scene.py        # the SI boundary: mag:<grade>:<dir> grammar, solid
                  #   classification, weld_mesh, the occurrence transform, and
                  #   load() -> MagScene
  kinematics.py   # the mate: sidecar selection by childId, delta(mate, q), the
                  #   posed/pathed source copies, cylindrical sub-DOFs
  physics.py      # getFT wrappers, projection, clearance, energy, equilibria,
                  #   the convergence protocol, the verdict, sweep() and field()
  report.py       # sweep.json and the plotly report/field HTML
```

The CLI verb module (`cadgen.cli.magnetics`) is the command surface over these;
it holds only argparse and error mapping and lives with the other command
shells, not here.
