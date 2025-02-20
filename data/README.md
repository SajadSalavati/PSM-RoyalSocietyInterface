# Data

The raw CFD datasets are **not** stored in this repository (they are large binary
archives). They are published on Zenodo and referenced by the paper.

> **Zenodo:** a direct link will be added here once the dataset is public.

## Expected layout

The pipeline reads data from `data/raw/` using the paths derived in
`ExperimentConfig.data_path`, i.e. `data/raw/<model>/<bc_name>/<npz_filename>`:

```
data/
└── raw/
    ├── BC.csv                                  # inlet velocity/pressure waveforms
    ├── Base/
    │   ├── Mid/  rylScty_patientSpec_BaseModel_miduvw.npz
    │   └── Wall/ rylScty_patientSpec_BaseModel_wallPre&OSI&WSS.npz
    └── Sten/
        ├── Mid/  rylScty_patientSpec_StenosisModel_miduvw.npz
        └── Wall/ rylScty_patientSpec_StenosisModel_wallPre&OSI&WSS.npz
```

## Archive contents

| Region | Keys inside the `.npz`            |
| ------ | -------------------------------- |
| `Mid`  | `x, y, z, v_x, v_y, v_z`         |
| `Wall` | `x, y, z, pre, osi, wss`         |

Each field is stored as `(time, location)` and transposed to `(location, time)` on load.

## `BC.csv`

Columns:

- `t1`, `v` — time and value of the inlet **velocity** waveform
- `t2`, `P` — time and value of the inlet **pressure** waveform (may contain trailing NaNs)

Both waveforms are linearly interpolated onto `n_snapshots` uniform time points.

`data/raw/` is git-ignored; only this README is tracked.
