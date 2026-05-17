# NMR Spectra Analysis Model

Browser-based showcase application for a trained **¹H + ¹³C NMR fusion model**.

The project demonstrates a machine learning workflow for analysing NMR spectra and predicting which compounds are likely present in a sample. The final model uses both **¹H NMR** and **¹³C NMR** spectra at the same time and supports both **single-compound prediction** and **multi-label mixture prediction**.

---

## Project overview

The application provides a simple local web interface for testing a trained neural network model. Instead of running predictions manually inside a notebook, the user can start the app, upload spectra, and inspect the model output directly in the browser.

The workflow is:

```text
¹H spectrum + ¹³C spectrum
        ↓
preprocessing and normalization
        ↓
dual-input fusion CNN model
        ↓
compound probabilities + predicted component count
        ↓
application output
```

The model combines information from both nuclei because **¹H and ¹³C spectra provide complementary information** about the analysed compound or mixture.

---

## What the app does

The browser app:

* automatically loads the trained model export `nmr_artifacts_fusion.zip`,
* accepts ¹H and ¹³C input files,
* supports `.npy`, CSV, TXT, XY spectra, and peak-list style files,
* converts the uploaded spectra into the required vector format,
* normalizes and plots the prepared spectra,
* runs prediction using the trained fusion model,
* predicts the number of components in the sample,
* displays detected compounds and top compound probabilities,
* optionally compares predictions with expected labels,
* exports prepared vectors and prediction results.

---

## Model architecture

The model is a **dual-input convolutional neural network**.

```text
¹H spectrum  ──► ¹H CNN branch ──┐
                                 ├──► fusion layers ──► compound labels
¹³C spectrum ──► ¹³C CNN branch ─┘                    └──► component count
```

Each input spectrum is processed by a separate CNN branch. The extracted feature vectors are then concatenated and passed through shared dense fusion layers.

The model has two output heads:

1. **Label head**

   * multi-label compound prediction,
   * sigmoid output over known compounds,
   * supports mixtures containing multiple compounds.

2. **Count head**

   * predicts whether the sample contains 1, 2, or 3 components,
   * uses softmax output,
   * helps the app select the most likely number of predicted compounds.

The main architecture contains:

* 12 Conv1D layers in the ¹H branch,
* 12 Conv1D layers in the ¹³C branch,
* 2 shared dense fusion layers,
* 2 output dense layers.

---

## Evaluation summary

The final fusion model was evaluated on a synthetic held-out test set.

### Final fusion model metrics

| Metric          |  Value |
| --------------- | -----: |
| Micro F1        | 0.9430 |
| Macro F1        | 0.9558 |
| Samples F1      | 0.9389 |
| Micro Precision | 0.9430 |
| Micro Recall    | 0.9429 |
| Exact Match     | 89.45% |
| Hamming Loss    | 0.0041 |
| Count Accuracy  | 99.94% |

The most important result is that the model achieved **94.30% Micro F1**, **89.45% Exact Match**, and **99.94% component-count accuracy**.

### Performance by mixture size

| Components | Samples | Micro F1 | Exact Match | Hamming Loss |
| ---------: | ------: | -------: | ----------: | -----------: |
|          1 |   5,184 |   0.9256 |      92.53% |       0.0028 |
|          2 |   3,508 |   0.9486 |      90.28% |       0.0038 |
|          3 |   4,268 |   0.9469 |      85.03% |       0.0059 |

Exact match is a strict metric: for a multi-label mixture, the prediction is counted as correct only if the full set of predicted compounds exactly matches the true set.

### Comparison with individual models

| Model                 | Test Micro F1 | Test Exact Match | Test Micro Recall |
| --------------------- | ------------: | ---------------: | ----------------: |
| ¹H individual model   |        0.8189 |           70.99% |            0.8297 |
| ¹³C individual model  |        0.9243 |           86.23% |            0.9407 |
| ¹H + ¹³C fusion model |        0.9430 |           89.45% |            0.9429 |

The fusion model achieved the best overall performance by combining complementary information from ¹H and ¹³C spectra.

---

## Important limitation

The model can only predict compounds that are included in its training label map.

If the user uploads a spectrum of a compound that was not part of the training dataset, the model will still return the closest known compound from its label map. In that case, the prediction may not be chemically correct.

The current model was trained mainly on synthetic spectra. Real spectra can differ from synthetic training data because of noise, solvent peaks, baseline shifts, missing peaks, or different experimental conditions. This domain shift is an important area for future improvement.

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

The model ZIP is loaded automatically when the app starts.

---

## Installation and startup

### Ubuntu / Linux

Open a terminal in the project folder.

Make the Linux run script executable:

```bash
chmod +x run_linux.sh
```

Start the app:

```bash
./run_linux.sh
```

After the app starts, open this address in your browser:

```text
http://127.0.0.1:7860
```

If the browser does not open automatically, copy the address from the terminal and paste it into your browser manually.

---

### Windows

Double-click:

```text
run_windows.bat
```

The script will create the environment, install dependencies, and start the local browser app.

Then open:

```text
http://127.0.0.1:7860
```

---

## Input files

The fusion model requires two input files:

1. ¹H NMR input,
2. ¹³C NMR input.

Both files are uploaded in the browser interface.

Supported input formats include:

* `.npy` vector files,
* one-column CSV/TXT vector files,
* two-column XY spectrum CSV/TXT files,
* peak-list style CSV/TXT files.

The app converts the input data into the vector format expected by the model.

---

## Example usage

1. Start the app.
2. Upload a ¹H spectrum file.
3. Upload a ¹³C spectrum file.
4. Optionally enter expected labels for comparison.
5. Run prediction.
6. Inspect the detected compounds, top probabilities, and predicted component count.

Example output:

```text
Predicted number of components: 2
Detected compounds:
- aniline
- cumene
```

The output also includes ranked compound probabilities, which makes it easier to inspect near-misses and similar compounds.

---

## Documentation

Project documentation is prepared as a separate PDF file and includes:

* algorithm and theory,
* implementation description,
* installation and startup instructions,
* running examples,
* evaluation results,
* limitations and future work.

---

## Authors

* Richard Golian
* Jan Hlaváč

Faculty of Informatics, Masaryk University

---

## Repository

```text
https://github.com/golianr/NMR-Spectra_Analysis-Model
```

## Acknowledgements

ChatGPT by OpenAI was used as a writing and documentation assistant during the preparation of the README, presentation, and project documentation.
