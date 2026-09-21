---
name: magnetics
description: Analyze permanent-magnet mechanisms authored as cadgen CAD models — force, torque and potential energy along a cadgen-declared slider, revolute or cylindrical mate. Use when the user asks which way a magnet arrangement pushes a moving part, where it can rest (equilibria/detents), whether a detent is strong enough to hold against friction, or wants a `|B|` field slice. Reads magnets declared by the `mag:<grade>:<direction>` leaf-naming convention from a generated `.step` and its sidecar; needs the `cadgen[magnetics]` extra.
---

# Magnetics

Provenance: maintained in [earthtojake/text-to-cad](https://github.com/earthtojake/text-to-cad).
Use the installed local skill files as the runtime source of truth; the
repository link is only for provenance and release review.

Use this skill to answer "what does this permanent-magnet arrangement do"
before building it. The product is **force, torque and potential energy along
the mechanism's constrained degree of freedom** — a cadgen `slider`, `revolute`
or `cylindrical` mate. Field slices are the explanatory picture, not the answer.
The analysis is quasi-static (hard magnets only; no soft iron, coils or eddy
currents) and never predicts a settling point — it reports magnetic tendency,
admissible motion and candidate rests.

This skill is instruction-only. Its runtime lives in cadgen behind an extra, so
every fact comes from the `cadgen magnetics` verbs; install
`requirements.txt` (which pins `cadgen[magnetics]`) into the active project
environment first, or the first verb dies with a
`pip install "cadgen[magnetics]"` hint. Run `cadgen` from that environment
(`python -m cadgen.cli magnetics <verb>` is the PATH-independent equivalent);
`cadgen doctor <this skill's directory>` verifies the installed cadgen matches
this skill's pin, since docs drift silently on a mismatched install. Use
`cadgen magnetics <verb> --help` for the complete current interface.

## The Three Verbs

- **`inspect`** — list the magnets a model declares (ref, label, shape class,
  world polarization, position, orientation) and the mate it drives. Reach for
  it first, to confirm the tool sees the magnets and mate you expect before
  spending a sweep.
- **`sweep`** — sample the mate across its limits, project the magnetic
  force/torque onto the DOF, integrate potential energy, find equilibria and
  their stability, check clearance per pose, run the convergence protocol, and
  write `sweep.json` + `report.html`. This is the answer to "which way, and
  where does it rest".
- **`field`** — sample `|B|` and in-plane arrows on one plane at one pose and
  write `field.html`. The explanatory picture; use it to show *why* a sweep came
  out the way it did, not to decide anything.

```bash
cadgen magnetics inspect model.step
cadgen magnetics sweep model.step --mate travel --friction-N 0.15 --out out/sweep
cadgen magnetics field model.step --slice mate --at 0 --out out/field
```

`--mate` defaults to the model's sole mate; name it when the model has more than
one. `--out` defaults to `<step>-magnetics/` beside the STEP.

## Declaring Magnets (naming convention)

Until a first-class `magnetics=` sidecar slot lands, a leaf occurrence **is** a
magnet when its name matches `mag:<grade>:<direction>`. The tool iterates the
scene's leaves and never resolves a magnet by name, so duplicate names are
legal.

- Inside `compound_from_instances` each leaf is named by its **instance-name
  argument**, so the convention rides that argument:
  `compound_from_instances("row_a", [(cube, loc, "mag:N42:+z"), ...])`. For a
  standalone part, `label_shape(cube, "mag", "N42", "+z")` yields the same
  `mag:N42:+z` string.
- `<grade>` is a grade from the table in `references/magnetics.md` (its
  remanence `Br` is used as `|J|` in tesla: N42 → 1.30 T) or an explicit
  `Br=1.32T`.
- `<direction>` is the magnetization axis in the part's **local** frame:
  `+x -x +y -y +z -z`, or a comma-separated vector the tool normalizes.

A magnet that must later become a sidecar target gets a distinct plain `#label`
as well; the `mag:` names are never `#label` aliases.

Every solid inside a magnet leaf becomes its own magpylib source. Axis-aligned
and tilted cuboids map to `Cuboid`, right cylinders to `Cylinder`; anything else
(an oblique prism, a fillet) falls to a welded `TriangularMesh`. A leaf whose
solid will not weld shut (open or disconnected) is an error naming the leaf, not
a silent bad field. See `references/magnetics.md` for the frames, the grade
table and the full validation matrix.

## Reading a Verdict

`sweep` reports a **split verdict** at the `--at` pose (default: the lower
limit). Keep the three axes distinct — they answer different questions:

- **`tendency`** (`+axis` / `-axis` / `indeterminate`): which way the magnetic
  generalized force `Q` points along the mate at that pose. `indeterminate`
  means `|Q|` is smaller than the run's own uncertainty band — the sign is not
  resolved, so **refine before trusting it** (raise `--samples`, run
  `--convergence full`), do not report it as "no force".
- **`exceeds_threshold`** (`true` / `false` / `null`): whether `|Q|` beats the
  friction threshold you passed (`--friction-N` for a translation DOF,
  `--friction-Nm` for a rotation DOF). It is `null` when you gave no threshold,
  or when the uncertainty band straddles it — again, refine rather than round.
- **`admissible`** (`true` / `false`): whether that tendency can actually move
  the part — `false` when it points past a limit at an end-stop pose, or into a
  pose the clearance check marked inaccessible. A strong tendency into a wall
  moves nothing.

**Equilibria** are the candidate rests: sign changes of `Q` confirmed at both
pose-grid resolutions, `stable` iff `dQ/dq < 0` (a detent) and unstable
otherwise (a tipping point). A stable equilibrium is where a detent sits.

The **friction caveat**: rest under friction is a *band* around each stable
equilibrium, not a point — the part rests anywhere `|Q|` stays under the
threshold. `--friction-N`/`--friction-Nm` is a scalar threshold, not a friction
model; transverse load `|F⊥|` and torque are guide loads, never friction inputs.

## Convergence — Never Skip It

Forces come from meshed targets and finite-difference gradients, so every
verdict carries a convergence check that refines mesh, finite-difference step
and pose grid **separately**. A run reports `converged: true/false/null`:

- `false` still **exits 0** with a warning on stderr — the numbers are there but
  did not settle. Read `convergence.reasons` (`mesh`, `eps`, `pose_grid`) and
  rerun the named axis harder, or run `--convergence full`. Do not present a
  non-converged tendency as fact.
- `null` means `--convergence off` — you asked for the finest level once and no
  refinement, so there is no uncertainty estimate at all.

```bash
cadgen magnetics sweep model.step --mate travel --convergence full --out out/full
cadgen magnetics sweep model.step --samples 401 --converge-tol 0.01 --out out/fine
```

`--convergence quick` (the default) refines mesh and pose grid; `full` adds the
finite-difference step; `off` skips refinement. Default `--samples` is 201.

## Cylindrical Mates and the Yaw Question

A `cylindrical` mate has two sub-DOFs, `<mate>.travel` (mm) and `<mate>.turn`
(degrees). Pick the swept one with `--dof` (required for this kind). Hold the
other with `--hold <value>`, or `--relax` it — set it at every swept pose to
where its own generalized force vanishes. A pose whose held DOF has no zero in
range is marked `wedged` (reported and hatched, never an error). This makes the
"does the guide permit yaw" question computable: sweep travel and relax turn.

```bash
cadgen magnetics sweep model.step --dof travel.travel --relax --out out/yaw
cadgen magnetics sweep model.step --dof travel.turn --hold 0 --out out/turn
```

## Field Slices

```bash
cadgen magnetics field model.step --slice mate --at 0 --out out/field
cadgen magnetics field model.step --slice z=5 --grid 120 --out out/z
```

`--slice mate` builds the plane containing the mate axis (`mate:z` fixes the
in-plane "up" axis); `x=<mm>|y=<mm>|z=<mm>` is an axis-normal plane. The output
overlays `|B|` as a heatmap, in-plane arrows and magnet outlines at the `--at`
pose. Use it to explain a sweep result, not to measure force.

## Workflow

1. Confirm the model declares magnets and a mate as expected:
   `cadgen magnetics inspect <model>.step`. Fix the CAD source (naming, mate)
   before sweeping if `inspect` sees the wrong shapes, polarizations or mate.
2. Sweep with the DOF's friction threshold if you have one:
   `--friction-N <newtons>` for a slider, `--friction-Nm <newton-metres>` for a
   revolute. For a cylindrical mate, pick the swept sub-DOF with `--dof`.
3. Read `converged` first. If it is `false`, refine the named axis and rerun
   before reporting anything.
4. Report the split verdict as three separate facts (tendency, exceeds_threshold,
   admissible), then the equilibria as candidate rests, stating that rest under
   friction is a band, not a point. Call an `indeterminate` tendency unresolved,
   not zero.
5. Offer a `field` slice at the verdict pose to show why. When `$cad` is
   installed, offer to flip a magnet's direction in the source and rerun to see
   the energy landscape change; when `$cad-viewer` is installed, hand it the
   STEP so the user can inspect the mechanism visually.

`report.html` opens offline (plotly embedded, ~3.5 MB); pass `--plotly cdn` for
the ~50 kB form that loads plotly from a CDN. `sweep.json` (`"schema":
"magnetics-report/1"`) carries every number the report draws; its full shape,
the grade table, frame conventions, convergence protocol and validation matrix
are in `references/magnetics.md`.
