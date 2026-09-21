# magnetic-channel models

Two rows of 45-degree-tilted 1/4" N42 cube magnets either side of a gap, with one
magnet sliding down the gap on a pin-slot mate. The magnetics sample the design
is built around; the polarity arrangement is the only thing that changes between
the first three models. Every model has **13 `mag:N42:<dir>` leaves** (two rows
of six plus the slider) and one mate. Shared geometry and kinematics live in
`lib/channel.py`.

| Script                   | Artifact                          | Arrangement                              | Mate                          |
|--------------------------|-----------------------------------|------------------------------------------|-------------------------------|
| channel_aligned.py       | STEP/channel_aligned.step         | every row magnet +z                      | `slider` `travel` along +x    |
| channel_alternating.py   | STEP/channel_alternating.step     | +z / -z along each row                    | `slider` `travel` along +x    |
| channel_one_flipped.py   | STEP/channel_one_flipped.step     | row_a middle magnet reversed (a defect)  | `slider` `travel` along +x    |
| channel_yaw.py           | STEP/channel_yaw.step             | aligned, but `travel` is `cylindrical`   | `cylindrical` travel + turn   |

Build (from the project root `models/magnetic-channel/`):

```bash
python src/channel_aligned.py
python src/channel_alternating.py
python src/channel_one_flipped.py
python src/channel_yaw.py
```

Unchanged models are no-ops. `lib/` is on the import path automatically when a
model is run by path from `src/`. STEP outputs are gitignored (see the project
`.gitignore`); the analysis commands are in `../README.md`.
