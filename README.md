# NMR Spectra Analysis Model

Machine learning pipeline for analyzing **¹H NMR spectra** and predicting chemical compounds from spectral data.

This repository focuses on training and evaluating machine learning models on NMR spectral vectors.

The dataset used in this project is synthetically generated using the companion repository:

**Synthetic dataset generator:**
https://github.com/golianr/SyntheticNmrGenerator

---

# Project Overview

The goal of this project is to develop a machine learning system capable of recognizing organic compounds from **1D ¹H NMR spectra**.

The project pipeline consists of two major components:

1. **Synthetic dataset generation**
   NMR spectra are generated using a custom synthetic spectrum generator.

2. **Spectral analysis model**
   A machine learning model is trained to classify compounds based on the generated spectral data.

---

# Pipeline Architecture

Synthetic dataset generation:

```
peak lists (ppm + intensity)
        ↓
Synthetic NMR generator
        ↓
vector spectra + PNG spectra
```

Spectral analysis pipeline:

```
NMR spectrum vector
        ↓
preprocessing
        ↓
machine learning model
        ↓
compound prediction
```

---

# Dataset

The dataset used for training is generated using the repository:

https://github.com/golianr/SyntheticNmrGenerator

The generator produces:

* synthetic **¹H NMR spectra vectors**
* corresponding **PNG spectral plots**
* metadata files linking spectra to compounds

Dataset structure example:

```
nmr_dataset/
    images/
        ethanol/
        acetone/
        toluene/
    vectors/
        ethanol/
        acetone/
        toluene/
    index.csv
    label_map.json
```

---

# Technologies Used

Python 3.x

Main libraries:

* NumPy
* pandas
* scikit-learn
* PyTorch
* torchvision
* matplotlib

---

# Model Goals

The primary objective of this repository is to:

* train machine learning models on NMR spectral vectors
* evaluate compound classification accuracy
* experiment with different neural network architectures

Potential future extensions include:

* mixture spectrum detection
* CNN models for spectrum images
* multi-label compound prediction
* integration with real experimental NMR data

---

# Future Work

Planned improvements:

* larger synthetic datasets
* improved spectral augmentation
* support for mixture spectra
* evaluation on real experimental NMR datasets

---

# Related Repository

Synthetic dataset generator used in this project:

https://github.com/golianr/SyntheticNmrGenerator

This generator creates the synthetic NMR spectra used to train and evaluate the models in this repository.

---

# Authors:

Richard Golian
