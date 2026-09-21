# Magnetics reference

The details behind `SKILL.md`: the naming grammar and grade table, frame
conventions, the convergence protocol, verdict semantics, the CLI validation
matrix, the report layout and the `sweep.json` schema. All quantities inside the
analysis are SI (metres, radians, tesla, newtons, newton-metres, joules); model
units (millimetres, degrees) enter only when a `.step` is loaded and are
converted once.

## Naming grammar

A leaf occurrence is a magnet when its name matches:

```
mag:<grade>:<direction>
```

- `<grade>`: a grade key from the table below, or an explicit remanence
  `Br=<value>T` (for example `Br=1.32T`). The remanence `Br` is used directly as
  the magnitude `|J|` of the polarization in tesla. Unknown grade tokens, a
  `Br=` without a `T` suffix, or `Br` ≤ 0 are errors that name the offending
  leaf and token.
- `<direction>`: the magnetization axis in the part's **local** frame. Either an
  axis token — `+x`, `-x`, `+y`, `-y`, `+z`, `-z` — or a comma-separated vector
  such as `0,0.7,0.7` that the tool normalizes. A zero-length vector is an error
  naming the leaf.

The label alias grammar is `[A-Za-z_][A-Za-z0-9_:]*`, so a `mag:...` name is
never a `#label` alias; a magnet that must become a sidecar target later carries
a distinct plain label as well. Names come from the instance-name argument of
`compound_from_instances`, or from `label_shape(part, "mag", "<grade>",
"<direction>")` for a standalone part. Duplicate names are legal — the tool
iterates `scene.leaves()` and never resolves a magnet by name.

## Grade table

Remanence `Br` (tesla), used as `|J|`:

| grade | Br (T) |
|---|---|
| N35 | 1.19 |
| N38 | 1.24 |
| N42 | 1.30 |
| N45 | 1.35 |
| N48 | 1.40 |
| N52 | 1.45 |
| SmCo26 | 1.05 |
| Y30 (ferrite) | 0.40 |

Anything not in the table must be given as `Br=<value>T`.

## Shape classification

Each solid inside a magnet leaf is classified from its **world-placed** faces,
independent of the others:

- **Cuboid**: six planar faces whose normals form three mutually orthogonal
  pairs, with rectangular faces. Maps to magpylib `Cuboid`.
- **Cylinder**: two parallel planar caps plus one cylindrical face whose axis is
  normal to the caps. Maps to magpylib `Cylinder`.
- **Mesh**: anything else (an oblique prism, a filleted body) is tessellated to
  a `TriangularMesh`. build123d's tessellation emits per-face vertex copies, so
  the tool welds vertices (rounds to 1e-6 m, merges, remaps faces, drops
  degenerate triangles) and constructs with the open/disconnected checks on. A
  leaf still open or disconnected after welding is an error naming the leaf and
  the failed check.

## Frames

magpylib's `polarization` and `dimension` are **body-frame** and are rotated by
the source's `orientation`; the tool never rotates twice.

- `R_occ` is the occurrence's local→world rotation, read through
  `Occurrence.world_transform` (a 4×4 world transform; the private
  `occ._node.transform` is the documented fallback). `J_world = |J| · R_occ ·
  dir_local`.
- **Cuboid / Cylinder**: the body frame `R_box` comes from the classified world
  face normals (cuboid: the orthogonal pair normals as right-handed columns;
  cylinder: the cap normal as body `z`). `dimension` is measured in that body
  frame, never from the world axis-aligned bounding box. `position` is the
  centroid, `orientation = R_box`, and `polarization = R_boxᵀ · J_world`.
- **Mesh**: world-space vertices, `orientation` identity, `polarization =
  J_world`.

`inspect` reports per magnet: `ref` (`#o1.2.3`), `label`, `shape`,
`polarization_world_T`, `position_m`, `orientation_quat` (body→world, scipy
`(x, y, z, w)` order) and `moving` (`null` unless a mate is selected).

## Kinematics and pose convention

The mate is read from the model's `<name>.step.json` sidecar through cadgen's
binding reader, so the sidecar schema version and its `documentHash` are checked
against the STEP bytes; a stale or mismatched pair is an error. A mate is
`{"name", "kind", "parent", "child", "childId", "axis": {"origin", "dir"},
"limits"}`, one of kind `slider`, `revolute` or `cylindrical`. The moving set is
`scene.resolve("#" + childId)` and its descendants — the authored `#label` is
never used for selection.

The pose convention matches the viewer's forward kinematics
(`kinematicsRuntime.js`): the artifact as written is `q = 0`, and
`world(q) = D(q) · world(0)` (premultiplied), with

- slider: `D = T(dir · q)`,
- revolute: `D = T(origin) · R(dir, q) · T(−origin)`,
- cylindrical: `D = T(dir · travel) · T(origin) · R(dir, turn) · T(−origin)`
  (the two commute — same axis).

Slider `q` is metres, revolute `q` is radians (cadgen stores degrees; the load
converts); `sweep.json` records the swept DOF's `q_unit`.

### Cylindrical sub-DOFs

A cylindrical mate has `<name>.travel` (metres) and `<name>.turn` (radians),
each with its own limits. `--dof <name>.travel|<name>.turn` picks the swept one
(required for this kind). The other sub-DOF is either held at `--hold <value>`
(default 0, must lie within its limits) or, with `--relax`, set at each swept
pose to where its own generalized force vanishes (`Q_turn = τ·â` for a held
turn, `Q_travel = F·â` for a held travel), found by bracketed bisection over its
limits. A pose whose held DOF has no zero in range is `wedged`: reported,
excluded from equilibrium search, hatched in the report — never an error.
`sweep.json` records `mate.dof`, `mate.held` and a per-sample `held_value`.

## Mechanics

- Force and torque come from magpylib `getFT` with `squeeze=False` so the result
  keeps its `(pose, target, 3)` shape, reduced by an explicit sum over targets.
- Slider: `Q = F·â`, transverse load `F⊥ = F − Q·â`. Revolute: `Q = τ·â`.
- **Clearance (per pose):** the moving solids at `D(q)` are tested for
  interference (`overlap_volume`, behind a bounding-box prefilter) against every
  fixed solid. A pose with overlap above 1e-3 mm³ is `inaccessible`: excluded
  from equilibrium search, energy integrated only per accessible interval, and
  hatched in the report.
- **Energy:** `U(q) − U(q₀) = −∫ Q dq` by trapezoid within each accessible
  interval.
- **Equilibria:** sign changes of `Q` present at *both* pose-grid resolutions,
  refined by bisection on fresh single-pose force calls; `stable` iff
  `dQ/dq < 0`.

## Convergence protocol

At the verdict pose and at each equilibrium, three axes refine independently:

- **mesh** ∈ {20, 50, 100} at finite-difference step `eps = 1e-5 m` → `ΔQ_mesh`.
- **eps** ∈ {1e-5, 1e-6} at mesh 100 → `ΔQ_eps`.
- **pose grid**: the finest level run at `N` and `2N − 1` nested poses →
  `ΔQ_grid`, plus equilibrium count/ordering agreement and `ΔU_max`.

`uncertainty = max(ΔQ_mesh, ΔQ_eps, ΔQ_grid)`. `converged` is `true` when each
relative delta (guarded by `Q_scale = 0.02 · max|Q|`) is under `--converge-tol`
(default 0.02), equilibria agree within 1 % of range, and `ΔU_max` is under 2 %
of the energy span; otherwise `false` with `reasons` naming the failed axes
(`"mesh"`, `"eps"`, `"pose_grid"`). `--convergence`:

- `quick` (default): mesh {20, 50} and the pose grid.
- `full`: all three axes.
- `off`: the finest level once, `converged: null` (no uncertainty estimate).

Transverse load and torque carry the same band. A non-converged run still exits
0 with a warning on stderr.

## Verdict semantics

At `--at q` (default `limits[0]`):

- **`tendency`**: `+axis`, `-axis`, or `indeterminate` when `|Q| < uncertainty`.
- **`exceeds_threshold`**: `true`/`false` against `--friction-N` /
  `--friction-Nm`; `null` when no threshold is given or when the band
  `[|Q| − u, |Q| + u]` straddles it.
- **`admissible`**: `false` when the tendency points outside `[lo, hi]` at a
  limit pose, or into an inaccessible interval; `true` otherwise.

Rest under friction is a band around each stable equilibrium, not a point.
`|F⊥|` and torque are guide loads, never friction inputs.

## CLI validation matrix

Exit 0 = analysis ran (including `converged: false`, warned on stderr);
1 = analysis error (message on stderr); 2 = usage error.

| condition | message names | exit |
|---|---|---|
| magnetics extra not installed | `pip install "cadgen[magnetics]"` | 1 |
| no leaf matches `mag:` | the STEP file | 1 |
| unknown grade token, `Br=` without `T` or ≤ 0 | the leaf ref and token | 1 |
| direction vector has zero length | the leaf ref | 1 |
| leaf solid fails weld checks (open / disconnected) | the leaf ref and which check | 1 |
| sidecar missing, stale (`documentHash` mismatch) or old schema | cadgen's binding error | 1 |
| `--mate` names no mate, or model has 0 or 2+ mates and none given | the available mate names | 1 |
| mate kind not slider, revolute or cylindrical | the mate and kind | 1 |
| mate lacks `childId` | the mate | 1 |
| moving subtree contains no magnets | the mate and child ref | 1 |
| `limits.value` reversed or `lo == hi` | the mate and limits | 1 |
| slice plane misses the model bounding box | the plane | 1 |
| every sampled pose is inaccessible (clearance) | the mate | 1 |
| cylindrical mate without `--dof`, or `--dof` naming a sub-DOF the mate lacks | the mate and its sub-DOFs | 2 |
| `--hold` outside the held sub-DOF's limits | the sub-DOF and limits | 2 |
| `--at` outside limits, `--samples < 3`, `--grid < 8`, bad `--slice` or `--dof` syntax | the flag and value | 2 |
| `--relax` finds no zero in range at a pose | (not an error: pose marked `wedged`) | 0 |

## Report layout

`report.html` (plotly; `include_plotlyjs=True` by default so it opens offline at
~3.5 MB, `--plotly cdn` for the ~50 kB CDN form), top to bottom:

1. Verdict banner: tendency, `Q ± u`, threshold and `exceeds_threshold`,
   `admissible`, equilibria, convergence status and reasons; states that rest
   under friction is a band.
2. `Q(q)` and `U(q)` with equilibria marked, inaccessible and wedged intervals
   hatched, and the `--at` pose marked.
3. Field slice at the `--at` pose (`|B|` heatmap, in-plane arrows, magnet
   outlines).
4. Magnets table.
5. Convergence table (axis, levels, Δ).

`field` writes a standalone `field.html` with the same slice.

## `sweep.json` schema (`magnetics-report/1`)

```json
{
  "schema": "magnetics-report/1",
  "units": {"length": "m", "force": "N", "torque": "N m", "energy": "J", "B": "T"},
  "q_unit": "m",
  "magnets": [{"ref": "#o1.2.1", "label": "mag:N42:+z", "shape": "cuboid",
               "polarization_world_T": [0, 0.92, 0.92], "position_m": [0, 0, 0],
               "orientation_quat": [0, 0, 0, 1], "moving": false}],
  "mate": {"name": "travel", "kind": "slider", "axis": [1, 0, 0], "origin_m": [0, 0, 0],
           "limits": [0, 0.08]},
  "samples": [{"q": 0.0, "F_N": [0.42, 0.1, 0.9], "Q": 0.42, "F_transverse_N": [0, 0.1, 0.9],
               "torque_Nm": [0, 0.0021, 0], "U_J": 0.0, "accessible": true}],
  "inaccessible": [[0.071, 0.08]],
  "wedged": [],
  "equilibria": [{"q": 0.0613, "stable": true, "dQdq": -12.1, "uncertainty_q": 0.0002}],
  "verdict": {"at_q": 0.0, "tendency": "+axis", "Q": 0.42, "uncertainty": 0.006,
              "threshold": 0.15, "exceeds_threshold": true, "admissible": true},
  "convergence": {"mesh": {"levels": [20, 50, 100], "eps": 1e-5, "dQ": 0.005},
                  "eps": {"levels": [1e-5, 1e-6], "meshing": 100, "dQ": 0.002},
                  "pose_grid": {"samples": [201, 401], "dQ": 0.001, "equilibria_agree": true, "dU_max": 0.004},
                  "converged": true, "reasons": []}
}
```

`dof`, `held` and per-sample `held_value` appear only for a cylindrical sweep;
`q_unit` is the swept DOF's unit, and `mate.limits` is that DOF's `[lo, hi]`.
`--json` on stdout is one compact line (no indentation); the pretty form above
is only for reading here.
