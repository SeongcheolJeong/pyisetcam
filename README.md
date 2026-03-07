# pyisetcam

`pyisetcam` is a greenfield Python port of the core ISETCam numerical pipeline:

- `scene -> optical image -> sensor -> image processor -> camera`
- pinned to upstream ISETCam commit `412b9f9bdb3262f2552b96f0e769b5ad6cdff821`
- validated through a GNU Octave parity harness for curated cases

## Status

This repository implements milestone-one scope:

- explicit-object APIs only, no `vcSESSION`
- core scene/display/oi/sensor/ip/camera objects and functions
- upstream asset fetching into `.cache/upstream/isetcam/<sha>/`
- unit tests and a parity harness scaffold

Out of scope for this milestone:

- GUI/App Designer windows
- ray-trace optics
- OpenEXR/MEX integrations
- facetracker, printing, and most peripheral modules

## Quickstart

Install Miniforge and create the environment:

```bash
conda env create -f environment.yml
conda activate isetcam-py
python -m pip install -e .
python tools/fetch_upstream.py
pytest
```

Run the example pipeline:

```bash
python examples/end_to_end.py
```

## Layout

- `src/pyisetcam/`: package source
- `tools/fetch_upstream.py`: fetch/extract the pinned upstream snapshot
- `tools/octave/`: Octave parity export helpers
- `tests/unit/`: deterministic unit tests
- `tests/parity/`: curated parity case definitions and baseline comparator
- `docs/migration.md`: MATLAB-to-Python name mapping

