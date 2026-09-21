# magnetic-channel

The sample the magnetics capability is built around: two rows of 45-degree-tilted
1/4" (6.35 mm) N42 cube magnets either side of a channel, with one magnet sliding
down the channel on a pin-slot mate. The question is whether the slider is pushed
toward one end, sits at an equilibrium, or is free to wander — and, for the yaw
variant, whether the pin-slot's rotational freedom matters.

Analysis needs the magnetics extra:

```bash
pip install "cadgen[magnetics]"
```

## Models

Four models, all built from `src/lib/channel.py`; see `src/README.md` for the
build commands and the catalog. Each has 13 `mag:N42:<dir>` leaves (two rows of
six plus the slider) and one mate named `travel`.

| STEP                              | Arrangement                             | Mate          |
|-----------------------------------|-----------------------------------------|---------------|
| `STEP/channel_aligned.step`       | every row magnet +z                     | slider        |
| `STEP/channel_alternating.step`   | +z / -z along each row                    | slider        |
| `STEP/channel_one_flipped.step`   | row_a middle magnet reversed (a defect) | slider        |
| `STEP/channel_yaw.step`           | aligned, travel + roll                   | cylindrical   |

Build them first (from this directory):

```bash
python src/channel_aligned.py
python src/channel_alternating.py
python src/channel_one_flipped.py
python src/channel_yaw.py
```

## Inspect

List the magnets and the mate a model declares:

```bash
cadgen magnetics inspect STEP/channel_aligned.step
```

## Sweep the three arrangements

Sweep the slider across the channel, project the magnetic force onto the travel
DOF, integrate the potential energy, find equilibria and read the verdict. The
three arrangements should differ — that is the point of the sample:

```bash
cadgen magnetics sweep STEP/channel_aligned.step     --mate travel --friction-N 0.15 --out tmp/aligned
cadgen magnetics sweep STEP/channel_alternating.step --mate travel --friction-N 0.15 --out tmp/alternating
cadgen magnetics sweep STEP/channel_one_flipped.step --mate travel --friction-N 0.15 --out tmp/one_flipped
```

Each writes `sweep.json` and `report.html` to its `--out` directory.
`--friction-N 0.15` is the static-friction force the verdict compares `|Q|`
against; drop it to read the raw tendency. For a publication-quality convergence
run:

```bash
cadgen magnetics sweep STEP/channel_aligned.step --samples 401 --convergence full --out tmp/aligned-full
```

## Field slice

Sample `|B|` and the in-plane field arrows on the plane containing the mate axis,
at the rest pose:

```bash
cadgen magnetics field STEP/channel_aligned.step --slice mate --at 0 --out tmp/aligned-field
```

## The pin-slot yaw question

`channel_yaw.step` puts the slider on a `cylindrical` mate — it can travel along
+x and roll about that axis. Sweep the travel and, at each pose, relax the roll
to where its own generalized torque vanishes; the relaxed `held_value` column is
the yaw curve:

```bash
cadgen magnetics sweep STEP/channel_yaw.step --mate travel --dof travel.travel --relax --out tmp/yaw
```

A pose whose roll has no torque zero in range is reported `wedged`, not an error.

## Bench validation (Phase 3)

Build one channel, measure the slider force at two poses with a force gauge, and
record the readings here against the swept `Q(q)`. Confirm whether the physical
pin-slot permits the roll the `--relax` curve assumes.

| pose (mm) | measured force (N) | swept `Q` (N) | notes |
|-----------|--------------------|---------------|-------|
| _tbd_     | _tbd_              | _tbd_         |       |
| _tbd_     | _tbd_              | _tbd_         |       |
