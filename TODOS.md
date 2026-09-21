# TODOS

## Magnetics

### `magnetics=` slot and `magnets` sidecar section

**What:** Add `@step(magnetics={...})` in `packages/cadgen/src/cadgen/authoring.py` and a resolved `magnets` section in `<name>.step.json` (schema 10; `_SIDECAR_SECTIONS`, `_WARRANTING_SECTIONS`, JS `SIDECAR_KEYS`), replacing the `mag:<grade>:<direction>` naming convention the first magnetics PR ships with.

**Why:** Physics data in name strings has no schema, no units, and cannot target duplicate labels. The sidecar is where cadgen keeps every other declaration (kinematics, appearance, animation).

**Context:** `cadgen magnetics sweep` writes a `magnets` array in `sweep.json` already shaped as the future section (`ref`, `label`, `shape`, `polarization_world_T`, `position_m`, `orientation_quat`). Mirror `normalize_materials` / `apply_appearance` in `_internal/source_sidecar.py` for the authoring form `{"definitions": {"N42": {"remanence_T": 1.30}}, "assignments": [{"targets": [...], "material": "N42", "magnetization": [0,0,1]}]}` stored resolved to numeric occurrence ids; then `authoring.py`; then `packages/cadgen-js/src/common/sourceSidecar.js`. A schema bump invalidates every existing sidecar until its model is rebuilt, so land it as its own PR with the repo owner's design review. Targets use the existing `#label` / `#o1.2.3` forms; globs would be new selector syntax.

**Effort:** M
**Priority:** P2
**Depends on:** the `cadgen.magnetics` PR landing.

### Sensitivity study over gap, tilt and remanence

**What:** A `cadgen magnetics sensitivity` mode that perturbs remanence (±5 % Br), gap (±0.5 mm) and tilt (±2°) and reports the spread of `Q` and of equilibrium positions.

**Why:** Nominal grade values and perfect alignment overstate confidence. magpylib documents magnetization and alignment variation as the dominant real-world error; a bench comparison needs "0.42 ± 0.06 N", not "0.42 N".

**Context:** Implement as a perturbation loop over `MagScene` copies in `cadgen/magnetics/physics.py`, reusing the vectorised sweep; add a `sensitivity` block to `sweep.json` and a band around the `Q` curve in the report. Do it after the first bench measurement, when the parameter the real channel is most sensitive to is known.

**Effort:** M
**Priority:** P3
**Depends on:** the `cadgen.magnetics` PR; one bench measurement of the channel.
