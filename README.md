# NMR Spectra Analysis Model

Browser showcase app for a trained **¹H + ¹³C NMR fusion model**.

The app runs locally in Python and opens a web interface in your browser. It supports Ubuntu/Linux and Windows through the same codebase.

## What the app does

- loads a fixed model export named `nmr_artifacts_fusion.zip`
- accepts ¹H and ¹³C inputs as `.npy`, vector CSV/TXT, XY spectrum CSV/TXT, or peak-list CSV/TXT
- preprocesses inputs to the model vector length
- plots both prepared spectra
- predicts compounds using the fusion model
- shows detected compounds, top probabilities, and optional expected-label check
- exports prepared vectors and prediction CSV files

## Required model file

Put your trained Colab export ZIP into the project root and name it exactly:

```text
nmr_artifacts_fusion.zip
```

Expected structure:

```text
NMR-Spectra_Analysis-Model/
├── app.py
├── requirements.txt
├── run_linux.sh
├── run_windows.bat
├── nmr_artifacts_fusion.zip   <-- add this manually
└── examples/
```

The ZIP should contain at least:

```text
*.keras
label_map.json
thresholds.json or similar threshold/config file
config.json or model_config.json
```

## Ubuntu / Linux

```bash
chmod +x run_linux.sh
./run_linux.sh
```

Then open:

```text
http://127.0.0.1:7860
```

The script creates a fresh `.venv` automatically if it does not exist.

## Windows

Double-click:

```text
run_windows.bat
```

Then open:

```text
http://127.0.0.1:7860
```

## Input formats

### `.npy` vector

A 1D NumPy array. If its length does not match the model input length, it is interpolated.

### Vector CSV/TXT

```csv
intensity
0.0
0.02
0.15
```

### XY spectrum CSV/TXT

```csv
ppm,intensity
7.26,0.8
7.25,1.0
```

### Peak list CSV/TXT

For ¹H, include multiplicity metadata when possible:

```csv
ppm,intensity,n_neighbors,j_hz_typical,exchangeable
3.66,0.667,3,7.1,False
1.18,1.000,2,7.1,False
```

`n_neighbors=3` generates a quartet. `n_neighbors=2` generates a triplet.

## Example ethanol files

The `examples/` directory contains small ethanol demo peak lists. The training-compatible ¹H example includes multiplicity information and is preferred over center-only peak lists.

## Notes

The model can only predict compounds that exist in its `label_map.json`. If the real spectrum is outside the trained label set, the app will choose the closest known class.
