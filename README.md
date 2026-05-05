# NMR Spectra Analysis Model

Browser showcase application for a trained **¹H + ¹³C NMR fusion model**.

This project demonstrates a machine learning model for analysing NMR spectra.  
The model uses both **¹H NMR** and **¹³C NMR** spectra at the same time and predicts which compounds are most likely present in the sample.

The application runs locally in Python and opens a simple web interface in your browser.  
It supports both **Ubuntu/Linux** and **Windows** using the same codebase.

---

## What this project is

This project is a showcase interface for a trained neural network model that analyses NMR spectra.

The model was trained on synthetically generated NMR spectra.  
Each sample contains:

- one **¹H NMR spectrum**
- one **¹³C NMR spectrum**
- one or more compound labels

The model supports both:

- **single-compound prediction**
- **multi-label mixture prediction**

This means the model can predict one compound or multiple compounds in a mixture.

---

## What the app does

The browser app:

- automatically loads a fixed model export named `nmr_artifacts_fusion.zip`
- accepts ¹H and ¹³C input files
- supports `.npy`, CSV, TXT, XY spectrum files, and peak-list files
- preprocesses the uploaded spectra to the correct model input size
- plots both prepared spectra
- runs prediction using the trained fusion model
- predicts the number of components in the sample
- displays the detected compounds
- displays top compound probabilities
- optionally checks the result against expected labels
- exports prepared input vectors and prediction result CSV files

---

## How the model works

The model is a **fusion neural network**.

Instead of using only one NMR spectrum, it uses two inputs:

```text
¹H spectrum  ──► ¹H CNN branch
                         │
                         ├──► fusion layers ──► compound prediction
                         │
¹³C spectrum ──► ¹³C CNN branch
```

The model has two main tasks:

1. **Compound prediction**
   - multi-label classification
   - predicts which compounds are present
   - uses sigmoid outputs for all known compounds

2. **Component count prediction**
   - predicts how many compounds are present
   - for example: 1, 2, or 3 components

The app then uses the predicted component count to select the most likely compounds.

Example:

```text
Predicted number of components: 2

Top probabilities:
ethanol        0.97
acetone        0.91
toluene        0.12
benzene        0.08
```

The app will return:

```text
Detected compounds:
ethanol
acetone
```

This approach is useful for mixtures because the model first estimates how many compounds should be present and then selects the most probable labels.

---

## Important limitation

The model can only predict compounds that exist in its training label map.

The supported compounds are stored inside:

```text
label_map.json
```

which is included inside the model export ZIP.

If you upload a spectrum of a compound that was not part of the training dataset, the model will still return the closest known compound from its label map.  
That prediction may not be chemically correct.

---

## Required model file

The app expects the trained model export ZIP to be placed in the root of the project.

The file must be named exactly:

```text
nmr_artifacts_fusion.zip
```

Expected project structure:

```text
NMR-Spectra_Analysis-Model/
├── app.py
├── requirements.txt
├── run_linux.sh
├── run_windows.bat
├── nmr_artifacts_fusion.zip
└── examples/
```

The model ZIP is not selected manually in the interface.  
The app loads it automatically when it starts.

The ZIP should contain at least:

```text
*.keras
label_map.json
thresholds.json
config.json or model_config.json
```

Typical contents:

```text
model.keras
label_map.json
thresholds.json
model_config.json
metrics.csv
```

---

## How to run on Ubuntu / Linux

Open a terminal in the project folder.

First, make the Linux run script executable:

```bash
chmod +x run_linux.sh
```

Then start the app:

```bash
./run_linux.sh
```

The script will automatically:

1. create a virtual environment named `.venv`
2. install the required Python packages
3. start the local browser app

After the app starts, open this address in your browser:

```text
http://127.0.0.1:7860
```

If the browser does not open automatically, copy the address from the terminal and paste it into your browser manually.

---

## How to run on Windows

Double-click:

```text
run_windows.bat
```

The script will automatically:

1. create a virtual environment named `.venv`
2. install the required Python packages
3. start the local browser app

Then open:

```text
http://127.0.0.1:7860
```

---

## First-time setup

Before starting the app, make sure that:

1. Python is installed
2. the project folder contains `nmr_artifacts_fusion.zip`
3. the model ZIP is named exactly `nmr_artifacts_fusion.zip`
4. you are running the app from the project root folder

Correct:

```text
NMR-Spectra_Analysis-Model/nmr_artifacts_fusion.zip
```

Incorrect:

```text
NMR-Spectra_Analysis-Model/models/nmr_artifacts_fusion.zip
```

The app currently expects the ZIP directly in the project root.

---

## Input files

The fusion model requires two input files:

1. **¹H NMR input**
2. **¹³C NMR input**

Both files are uploaded in the browser interface.

After upload, the app:

1. reads the input data
2. converts it into a numeric vector
3. interpolates it to the model input length if needed
4. normalizes the signal
5. plots the processed spectrum
6. runs the neural network prediction

---

## Supported input formats

### 1. `.npy` vector

A `.npy` file should contain a one-dimensional NumPy array.

Example:

```text
sample_1H.npy
sample_13C.npy
```

This is the best format when you already have spectra saved as vectors.

If the vector length does not match the model input length, the app automatically interpolates it.

---

### 2. Vector CSV or TXT

A simple one-column file containing intensity values.

Example:

```csv
intensity
0.0
0.02
0.15
0.40
0.18
```

This format is useful if the spectrum is already converted into a vector, but saved as CSV or TXT instead of NumPy.

---

### 3. XY spectrum CSV or TXT

A two-column spectrum file.

The first column should be the x-axis, usually ppm.  
The second column should be the signal intensity.

Example:

```csv
ppm,intensity
7.26,0.8
7.25,1.0
7.24,0.7
```

This format is useful for exported spectra from NMR software or public spectral databases.

The app interpolates the XY spectrum to the required model vector length.

---

### 4. Peak list CSV or TXT

A peak list contains peak positions and relative intensities.

Simple example:

```csv
ppm,intensity
3.66,0.667
1.18,1.000
```

This format is supported, but for ¹H NMR it can be less accurate if it only contains peak centers.

For better ¹H predictions, include multiplicity information.

---

## Recommended ¹H peak list format

For ¹H NMR, the model works better when the peak list includes multiplet metadata.

Recommended format:

```csv
ppm,intensity,n_neighbors,j_hz_typical,exchangeable
3.66,0.667,3,7.1,False
1.18,1.000,2,7.1,False
```

Column meanings:

| Column | Meaning |
|---|---|
| `ppm` | Chemical shift in ppm |
| `intensity` | Relative peak intensity |
| `n_neighbors` | Number of neighbouring hydrogens |
| `j_hz_typical` | Typical J-coupling constant in Hz |
| `exchangeable` | Whether the proton is exchangeable, for example OH or NH |

Multiplicity examples:

| `n_neighbors` | Generated multiplet |
|---:|---|
| 0 | singlet |
| 1 | doublet |
| 2 | triplet |
| 3 | quartet |
| 4 | quintet |
| 5 | sextet |
| 6 | septet |

Example:

```text
n_neighbors = 3
```

generates a quartet.

```text
n_neighbors = 2
```

generates a triplet.

---

## Why center-only peak lists can fail

A center-only ¹H peak list contains only peak positions and intensities.

Example:

```csv
ppm,intensity
3.66,0.667
1.18,1.000
```

This means the app only sees simple peak centers.  
For ¹H NMR, this can make the input look like a set of singlets.

However, the model was trained on spectra that include multiplet structure.  
For example, ethanol is expected to have a quartet and a triplet in its ¹H spectrum.

Better ethanol example:

```csv
ppm,intensity,n_neighbors,j_hz_typical,exchangeable
3.66,0.667,3,7.1,False
1.18,1.000,2,7.1,False
```

This input is more similar to the synthetic data used during training.

Without multiplicity information, the model may confuse the compound with another molecule that has similar simple peak positions.

---

## Example ethanol files

The `examples/` folder contains small ethanol demo files.

Recommended files:

```text
examples/ethanol_1H_training_compatible_peaklist.csv
examples/ethanol_13C_training_compatible_peaklist.csv
```

The ¹H file includes multiplicity information and is preferred over center-only peak lists.

Recommended test procedure:

1. Start the app.
2. Upload the example ¹H ethanol file.
3. Upload the example ¹³C ethanol file.
4. Click the prediction button.
5. Check the plotted spectra.
6. Check the detected compounds and top probabilities.

---

## Expected-label check

The app includes an optional expected-label check.

You can type the expected compound name into the expected labels field.

Example:

```text
ethanol
```

For mixtures, separate labels with commas:

```text
ethanol, acetone
```

The app will compare the expected labels with the detected labels and show:

- whether the prediction was an exact match
- missed labels
- extra predicted labels
- rank of the expected compound in the probability list

Example:

```text
Expected: ethanol
Detected: ethanol
Exact match: true
```

If the expected compound is not detected but appears in the top probabilities, the app can still show its rank.

Example:

```text
ethanol: rank=3, p=0.0464
```

This means ethanol was not selected as a final detected compound, but it was the third highest model probability.

---

## Understanding the output

The app shows several outputs.

### Predicted number of components

This is the model estimate of how many compounds are present in the sample.

Example:

```text
Predicted n_components: 2
```

This means the model expects two compounds in the uploaded spectra.

### Detected compounds

These are the final predicted compounds.

Example:

```text
ethanol        0.9721
acetone        0.9185
```

The number is the model probability or score for that compound.

### Top probabilities

This table shows the highest probability compounds, even if they were not selected as final detections.

This is useful for debugging.  
If the correct compound is ranked highly but not selected, the model was close.  
If the correct compound is not in the top results, the input is likely too different from the training data or the compound is not supported.

---

## Exported files

After prediction, the app can export prepared vectors and prediction results.

Typical exported files:

```text
prepared_1H_vector.npy
prepared_13C_vector.npy
prediction_top_probabilities.csv
prediction_detected_compounds.csv
```

These files are useful for debugging, documentation, or comparing predictions between different spectra.

---

## Recommended workflow

For the best results:

1. Start with the example files in `examples/`.
2. Confirm that the model loads correctly.
3. Test simple known compounds first.
4. Use ¹H peak lists with multiplicity metadata whenever possible.
5. Use both ¹H and ¹³C inputs.
6. Check the top probabilities, not only the final detected labels.
7. For real spectra, remove or reduce obvious solvent/reference peaks when needed.
8. Remember that the model only knows compounds from its training label map.

---

## Real spectra limitations

The model was trained on synthetically generated spectra.

Real NMR spectra may differ because of:

- solvent peaks
- TMS or reference peaks
- noise
- baseline drift
- phase distortion
- different linewidths
- different intensity scaling
- impurities
- concentration differences
- temperature, pH, or solvent effects
- peak overlap in mixtures

Because of this, real spectra may be harder than synthetic test samples.

For real data, the model should be treated as a demonstration and decision-support tool, not as a guaranteed chemical identification system.

---

## Troubleshooting

### The app says the model file is missing

Make sure the file exists in the project root:

```text
nmr_artifacts_fusion.zip
```

The name must match exactly.

---

### The app opens, but prediction fails

Check that the uploaded files contain numeric data.

For CSV files, the app expects either:

```text
intensity
```

or:

```text
ppm,intensity
```

or:

```text
ppm,intensity,n_neighbors,j_hz_typical,exchangeable
```

---

### The model predicts the wrong compound

Possible reasons:

- the compound is not in `label_map.json`
- the real spectrum is too different from the synthetic training data
- the ¹H peak list does not include multiplicity information
- solvent or impurity peaks dominate the spectrum
- one of the required inputs, ¹H or ¹³C, is missing or incorrect
- the ppm axis or intensity scaling is different from expected

---

### Ethanol is not detected from a simple peak list

Use the training-compatible ethanol example:

```text
examples/ethanol_1H_training_compatible_peaklist.csv
examples/ethanol_13C_training_compatible_peaklist.csv
```

The ¹H file includes quartet and triplet metadata, which is closer to the training data.

---

## Notes

This application is intended as a showcase for an NMR spectra analysis model.

It is useful for:

- demonstrating ¹H + ¹³C fusion prediction
- testing synthetic NMR vectors
- testing peak-list inputs
- visualising model predictions
- showing multi-label mixture classification

The model should not be used as the only source of chemical identification for real laboratory samples.