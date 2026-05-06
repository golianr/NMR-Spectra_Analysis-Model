# Dataset-based NMR examples

These examples were generated directly from the uploaded `spectra_dataset.csv` rows.
They are designed to be closer to the training data than generic web peak lists.

## What is included

- `single_peaklists/`: single-compound 1H and 13C peak-list CSV files
- `mixture_peaklists/`: multi-label mixture 1H and 13C peak-list CSV files
- `single_vectors/`: pre-rendered `.npy` vectors for single examples
- `mixture_vectors/`: pre-rendered `.npy` vectors for mixture examples
- `dataset_based_examples_manifest.csv`: index with expected labels and file paths

## Recommended usage in the app

Use the peak-list CSV version first, because it keeps the human-readable NMR peaks:

```text
examples/dataset_based/mixture_peaklists/mix_ethanol_acetone_1H_from_spectra_dataset_peaklist.csv
examples/dataset_based/mixture_peaklists/mix_ethanol_acetone_13C_from_spectra_dataset_peaklist.csv
```

Expected labels:

```text
ethanol, acetone
```

If a mixture is still missed by the app, try the `.npy` vector version from `mixture_vectors/`.
Those vectors are already rendered from the same peak lists and avoid CSV/peak-list preprocessing differences.

## Why these should work better than web examples

Previous examples were mostly center-only peak lists. For 1H NMR, this removes multiplet information.
These files preserve the columns from your dataset, especially:

```text
n_neighbors
j_hz_typical
exchangeable
multiplicity_model
```

So ethanol is represented as a quartet + triplet instead of two singlet-like peaks.

## Recommended mixture tests

Start with these easier examples:

```text
mix_ethanol_acetone
mix_ethanol_ethyl_acetate
mix_hexane_toluene
mix_ethanol_acetone_ethyl_acetate
mix_diethyl_ether_acetone_ethanol
```

Harder examples are intentionally included too:

```text
mix_chloroform_dichloromethane
mix_dimethyl_sulfoxide_acetonitrile_nitromethane
mix_toluene_p_xylene_mesitylene
```

These contain compounds with very simple or very similar spectra and can still be difficult.

## Notes

- These are not raw measured real spectra.
- They are showcase/test inputs derived from your `spectra_dataset.csv`.
- Mixture intensities were made by scaling each component with a mixture weight.
- The model must contain these compound names in `label_map.json`.
- Do not enable common artifact/solvent peak removal when the target compound is chloroform.

Generated from: `spectra_dataset.csv`
Fusion compounds available in source dataset: 48
Generated examples: 38
